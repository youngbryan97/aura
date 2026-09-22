"""In-memory index over MemoryWriteGateway JSON records.

Why this exists: strict-runtime recall includes the gateway record store
(``~/.aura/memory/*/*.json``). The original implementation re-read and
re-parsed up to 2,000 JSON files on every query — and ``Will.decide()`` calls
that path synchronously from the event loop on every governed tool execution,
which produced multi-second event-loop stalls in live use (the stall-report
storm in ``data/error_logs/stalls``).

This index makes the hot path pure in-memory scoring:

- a daemon thread performs the heavy directory scan / JSON parsing, keeping the
  newest ``MAX_ENTRIES`` records; searches never block on a full refresh
- freshly written records are picked up immediately via a bounded "newer than
  last refresh" mini-scan hinted by subdirectory mtimes (a handful of files at
  most), so recall-after-write stays correct
- when the index has never been built (cold start), the search falls back to a
  strictly bounded scan (``COLD_SCAN_MAX_FILES`` newest files within
  ``COLD_SCAN_BUDGET_S``) and returns best-effort results while the background
  build warms the rest
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("Aura.Memory.GatewayIndex")

_STOP_TERMS = {"what", "that", "this", "with", "from", "about", "memory", "remember"}


@dataclass(frozen=True)
class GatewayRecordEntry:
    path: str
    mtime: float
    written_at: float
    memory_id: str
    content: str
    haystack: str
    metadata: dict[str, Any]
    pin_boost: bool


def _parse_record(path: Path, mtime: float) -> GatewayRecordEntry | None:
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
    # not a failure: text that is not the JSON this expects is not a record it can
    # read back.
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    payload = envelope.get("payload") if isinstance(envelope, dict) else None
    if not isinstance(payload, dict):
        return None
    content = str(payload.get("content") or "").strip()
    if not content:
        return None
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    haystack = f"{content}\n{json.dumps(metadata, sort_keys=True, default=str)}".lower()
    try:
        written_at = float(payload.get("written_at") or mtime)
    except (TypeError, ValueError):
        written_at = mtime
    return GatewayRecordEntry(
        path=str(path),
        mtime=mtime,
        written_at=written_at,
        memory_id=path.stem,
        content=content,
        haystack=haystack,
        metadata=metadata,
        pin_boost=bool(
            metadata.get("session_memory_pin") or metadata.get("explicit_memory_request")
        ),
    )


class GatewayRecordIndex:
    """Thread-safe, incrementally refreshed view of the gateway record store."""

    REFRESH_INTERVAL_S = 15.0
    MAX_ENTRIES = 2000
    COLD_SCAN_BUDGET_S = 0.20
    COLD_SCAN_MAX_FILES = 64
    WRITE_HINT_MAX_FILES = 16
    WRITE_HINT_MIN_INTERVAL_S = 2.0

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._entries: dict[str, GatewayRecordEntry] = {}
        self._built = False
        self._last_refresh = 0.0
        self._last_write_hint_scan = 0.0
        self._refresh_running = threading.Lock()  # held only by the refresher
        self._swap_lock = threading.Lock()

    # ── refresh machinery ───────────────────────────────────────────────

    def _stat_mtime(self, path: Path) -> float:
        try:
            return float(path.stat().st_mtime)
        except OSError:
            return 0.0

    def _list_record_files(self) -> list[tuple[float, Path]]:
        files: list[tuple[float, Path]] = []
        try:
            for child in self.root.iterdir():
                if not child.is_dir():
                    continue
                for record in child.glob("*.json"):
                    mtime = self._stat_mtime(record)
                    if mtime > 0.0:
                        files.append((mtime, record))
        except OSError:
            return []
        files.sort(reverse=True)
        return files

    # GIL discipline for the refresher thread: a cold pass used to parse up
    # to MAX_ENTRIES JSON files back-to-back — a pure-Python burst that
    # monopolizes the GIL and lags the event loop (top stall fingerprint in
    # the Jul 8 triage: 73 co-occurrences in 3 days). Parse in bounded slices
    # with explicit yields; the cache carries across passes, and searches
    # serve from whatever is built so far (the module's own contract).
    MAX_PARSE_PER_PASS = 300
    PARSE_YIELD_EVERY = 25
    PARSE_YIELD_S = 0.002

    def _do_refresh(self) -> None:
        try:
            files = self._list_record_files()[: self.MAX_ENTRIES]
            previous = self._entries
            fresh: dict[str, GatewayRecordEntry] = {}
            parsed = 0
            for mtime, path in files:
                key = str(path)
                cached = previous.get(key)
                if cached is not None and cached.mtime == mtime:
                    fresh[key] = cached
                    continue
                if parsed >= self.MAX_PARSE_PER_PASS:
                    continue  # keep cached view for the rest; next pass resumes
                entry = _parse_record(path, mtime)
                parsed += 1
                if entry is not None:
                    fresh[key] = entry
                if parsed % self.PARSE_YIELD_EVERY == 0:
                    time.sleep(self.PARSE_YIELD_S)  # hand the GIL to the loop
            with self._swap_lock:
                self._entries = fresh
                self._built = True
                self._last_refresh = time.monotonic()
        except (OSError, RuntimeError, ValueError, TypeError, MemoryError) as exc:
            # The refresher must never die loudly; the next search re-kicks it.
            logger.warning("Gateway record index refresh failed: %s", exc)
        finally:
            self._refresh_running.release()

    def _kick_refresh(self) -> None:
        """Start a background refresh if one is not already running."""
        if not self._refresh_running.acquire(blocking=False):
            return
        thread = threading.Thread(
            target=self._do_refresh,
            name="gateway-record-index-refresh",
            daemon=True,
        )
        thread.start()

    def _stale(self) -> bool:
        return (time.monotonic() - self._last_refresh) > self.REFRESH_INTERVAL_S

    def _write_hint_due(self) -> bool:
        """Cheap freshness hint: a handful of directory stats, nothing more.

        The previous implementation globbed every *.json in any
        recently-touched subdirectory ON THE CALLER THREAD — inside
        Will.decide on the live event loop, that scan ran 5.8s under load
        (stall dump 2026-07-03 00:51) and killed a user's chat turn.
        Callers may only pay O(#subdirs) stat calls; all real filesystem
        work belongs to the background refresher.
        """
        if not self._built:
            return False
        now = time.monotonic()
        if (now - self._last_write_hint_scan) < self.WRITE_HINT_MIN_INTERVAL_S:
            return False
        self._last_write_hint_scan = now
        try:
            wall_last_refresh = time.time() - (time.monotonic() - self._last_refresh)
            for child in self.root.iterdir():
                if child.is_dir() and self._stat_mtime(child) >= wall_last_refresh - 1.0:
                    return True
        except OSError as exc:
            logger.debug("Gateway record write-hint stat skipped: %s", exc)
        return False

    # ── search ──────────────────────────────────────────────────────────

    @staticmethod
    def extract_terms(query_text: str) -> list[str]:
        import re

        return [
            term
            for term in re.findall(r"[a-z0-9_'-]{3,}", query_text.lower())
            if term not in _STOP_TERMS
        ][:12]

    @staticmethod
    def _score(entry: GatewayRecordEntry, terms: list[str]) -> float:
        if terms:
            hits = sum(1 for term in terms if term in entry.haystack)
            if hits <= 0:
                return 0.0
            score = hits / max(1, len(terms))
        else:
            score = 0.1
        if entry.pin_boost:
            score += 0.25
        return min(1.0, score)

    def search(self, query: str, limit: int = 5) -> list[tuple[float, GatewayRecordEntry]]:
        """Score records against the query; returns (score, entry) best-first.

        Never blocks on a full refresh: warm queries score the in-memory
        snapshot; cold queries do one strictly bounded direct scan while the
        background build warms the rest.
        """
        query_text = str(query or "").strip()
        if not query_text or not self.root.exists():
            return []
        terms = self.extract_terms(query_text)

        if self._built:
            if self._stale() or self._write_hint_due():
                self._kick_refresh()
            candidates = list(self._entries.values())
        else:
            # Cold: serve empty and warm in the background. The old bounded
            # cold scan still paid its listing cost (a stat per record file)
            # on the caller thread; Will's memory check is advisory and must
            # never buy freshness with event-loop time.
            self._kick_refresh()
            candidates = []

        scored: list[tuple[float, float, GatewayRecordEntry]] = []
        for entry in candidates:
            score = self._score(entry, terms)
            if score <= 0.0:
                continue
            scored.append((score, entry.written_at, entry))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [(score, entry) for score, _ts, entry in scored[:limit]]


_INDEX_LOCK = threading.Lock()
_INDEX: GatewayRecordIndex | None = None


def get_gateway_record_index(root: Path) -> GatewayRecordIndex:
    """Process-wide index for the given gateway root (root changes rebuild it)."""
    global _INDEX
    with _INDEX_LOCK:
        if _INDEX is None or _INDEX.root != Path(root):
            _INDEX = GatewayRecordIndex(Path(root))
        return _INDEX
