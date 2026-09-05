"""Durable holding queue for memory writes that were DEFERRED, not refused.

A deferral is a "not now". Treating it as a "no" destroys work that was
already performed — the live 2026-07-25 instance spent 10.8 seconds on a
web search, extracted facts and citations, then dropped the whole artifact
because welfare recovery drive happened to be high at that instant, and
logged it as ``all memory backends rejected the artifact``. Nothing was
wrong with the knowledge; the Will simply said "later", and "later" never
came because nobody was holding it.

This queue holds those writes and replays them when the runtime is willing
again. It deliberately does NOT relax any gate:

* A **deferral** is queued and retried. The Will's authority is untouched —
  the write still has to be approved on some later attempt.
* A **refusal** on the merits (content, provenance, constitutional rule) is
  dropped immediately with a log line. Retrying a decided "no" would be a
  bypass by persistence, which is the failure mode this file must not have.
* Nothing is bounded only by good intentions: the queue caps entries, caps
  age, and dedups by content hash, so a runtime that stays unwilling for a
  week does not accumulate an unbounded backlog of stale artifacts.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.governance_context import local_internal_governed_scope
from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.errors import record_degradation
from core.runtime.flags import FlagKind, declare

logger = logging.getLogger("Aura.DeferredRetention")

_WRITE_SOURCE = "deferred_retention"

FLAG_ENABLED = declare(
    "AURA_DEFERRED_RETENTION",
    kind=FlagKind.BOOL,
    default=True,
    description=(
        "Hold memory writes that were deferred (not refused) and replay them "
        "when the runtime is willing, instead of discarding completed work."
    ),
    owner="core/memory/deferred_retention.py",
)

FLAG_PATH = declare(
    "AURA_DEFERRED_RETENTION_PATH",
    kind=FlagKind.STRING,
    default="",
    description="Override the deferred-retention queue file (tests and soaks).",
    owner="core/memory/deferred_retention.py",
)

MAX_ENTRIES = 256
MAX_AGE_S = 7 * 24 * 3600.0
MAX_ATTEMPTS = 12

# Substrings that mark a "not now" rather than a decided "no". A reason that
# matches none of these is treated as a refusal on the merits and dropped —
# fail-closed, because the dangerous direction here is retrying a real veto.
_DEFERRAL_MARKERS = (
    "defer",
    "aura_now",
    "recovery_required",
    "stabilization",
    "not_ready",
    "backpressure",
    "resource_busy",
    "gate_unavailable",
    "try_again",
    "temporarily",
)

# Reasons that are decided refusals even though they contain a marker above.
_REFUSAL_OVERRIDES = (
    "constitutional_violation",
    "identity_protection",
    "content_rejected",
    "provenance_missing",
    "forbidden",
)


def is_deferral(reason: Any) -> bool:
    """Return whether a rejection reason means "later" rather than "no"."""

    text = str(reason or "").strip().lower()
    if not text:
        return False
    if any(marker in text for marker in _REFUSAL_OVERRIDES):
        return False
    return any(marker in text for marker in _DEFERRAL_MARKERS)


def _default_queue_path() -> Path:
    override = FLAG_PATH.value()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".aura/data/memory/deferred-retention.jsonl"


@dataclass(frozen=True)
class ReplayReport:
    """What one replay pass actually did — every count is observed, not assumed."""

    committed: int = 0
    still_deferred: int = 0
    refused: int = 0
    expired: int = 0
    remaining: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "committed": self.committed,
            "still_deferred": self.still_deferred,
            "refused": self.refused,
            "expired": self.expired,
            "remaining": self.remaining,
        }

    def narrative(self) -> str:
        if not any(
            (self.committed, self.still_deferred, self.refused, self.expired)
        ):
            return "no deferred memory writes were waiting"
        parts: list[str] = []
        if self.committed:
            parts.append(f"{self.committed} write(s) finally landed")
        if self.still_deferred:
            parts.append(f"{self.still_deferred} still deferred")
        if self.refused:
            parts.append(f"{self.refused} refused on the merits and dropped")
        if self.expired:
            parts.append(f"{self.expired} aged out")
        return "; ".join(parts)


class DeferredRetentionQueue:
    """Durable, bounded, deduped holding area for deferred memory writes."""

    def __init__(self, queue_path: Path | None = None) -> None:
        self._queue_path = (
            Path(queue_path).expanduser()
            if queue_path is not None
            else _default_queue_path()
        )

    @property
    def path(self) -> Path:
        return self._queue_path

    # ── read/write ────────────────────────────────────────────────────────

    def _load(self) -> list[dict[str, Any]]:
        if not self._queue_path.exists():
            return []
        try:
            raw = self._queue_path.read_text(encoding="utf-8")
        except OSError as exc:
            record_degradation(
                "deferred_retention", exc, severity="warning",
                action="deferred memory writes could not be read back",
            )
            return []
        entries: list[dict[str, Any]] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn line loses one write, never the whole queue
            if isinstance(record, dict) and record.get("text"):
                entries.append(record)
        return entries

    def _serialize(self, entries: list[dict[str, Any]]) -> str:
        return "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries)

    def _prepare(self, entries: list[dict[str, Any]]) -> str:
        return self._serialize(entries[-MAX_ENTRIES:])

    def _store(self, entries: list[dict[str, Any]]) -> None:
        gateway = get_file_write_gateway()
        with local_internal_governed_scope("deferred_retention_queue"):
            gateway.ensure_directory(self._queue_path.parent, source=_WRITE_SOURCE)
            gateway.write_text(
                self._queue_path, self._prepare(entries), source=_WRITE_SOURCE
            )

    async def _store_async(self, entries: list[dict[str, Any]]) -> None:
        gateway = get_file_write_gateway()
        payload = self._prepare(entries)
        with local_internal_governed_scope("deferred_retention_queue"):
            await gateway.ensure_directory_async(
                self._queue_path.parent, source=_WRITE_SOURCE
            )
            await gateway.write_text_async(
                self._queue_path, payload, source=_WRITE_SOURCE
            )

    # ── public API ────────────────────────────────────────────────────────

    def pending(self) -> list[dict[str, Any]]:
        return self._load()

    def _build_entry(
        self,
        text: str,
        metadata: Mapping[str, Any] | None,
        *,
        reason: Any,
        origin: str,
    ) -> dict[str, Any] | None:
        if not FLAG_ENABLED.value():
            return None
        body = str(text or "").strip()
        if not body:
            return None
        if not is_deferral(reason):
            logger.info(
                "Deferred-retention declined to hold a write refused on the "
                "merits (%s) from %s — a decided no is not retried.",
                reason, origin,
            )
            return None
        return {
            "hash": hashlib.sha256(body.encode("utf-8")).hexdigest()[:32],
            "text": body,
            "metadata": dict(metadata or {}),
            "origin": str(origin),
            "reason": str(reason),
            "queued_at": time.time(),
            "attempts": 0,
        }

    def _merge(
        self, entries: list[dict[str, Any]], entry: dict[str, Any]
    ) -> list[dict[str, Any]] | None:
        """Return the new entry list, or None when it is already held."""

        for existing in entries:
            if existing.get("hash") == entry["hash"]:
                return None
        return [*entries, entry]

    def enqueue(
        self,
        text: str,
        metadata: Mapping[str, Any] | None = None,
        *,
        reason: Any,
        origin: str = "unknown",
    ) -> bool:
        """Hold a deferred write. Returns whether it is now held."""

        entry = self._build_entry(text, metadata, reason=reason, origin=origin)
        if entry is None:
            return False
        try:
            merged = self._merge(self._load(), entry)
            if merged is None:
                return True  # already waiting; holding it twice helps nobody
            self._store(merged)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "deferred_retention", exc, severity="warning",
                action="a deferred memory write could not be held for retry",
                extra={"origin": origin},
            )
            return False
        logger.info(
            "Holding a deferred memory write from %s for retry (%s).",
            origin, entry["reason"],
        )
        return True

    async def enqueue_async(
        self,
        text: str,
        metadata: Mapping[str, Any] | None = None,
        *,
        reason: Any,
        origin: str = "unknown",
    ) -> bool:
        entry = self._build_entry(text, metadata, reason=reason, origin=origin)
        if entry is None:
            return False
        try:
            merged = self._merge(self._load(), entry)
            if merged is None:
                return True
            await self._store_async(merged)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "deferred_retention", exc, severity="warning",
                action="a deferred memory write could not be held for retry",
                extra={"origin": origin},
            )
            return False
        logger.info(
            "Holding a deferred memory write from %s for retry (%s).",
            origin, entry["reason"],
        )
        return True

    async def replay(self, memory_facade: Any = None) -> ReplayReport:
        """Retry every held write once. The gate still decides each one."""

        entries = self._load()
        if not entries:
            return ReplayReport()

        if memory_facade is None:
            from core.container import ServiceContainer

            memory_facade = ServiceContainer.get("memory_facade", default=None)
        if memory_facade is None or not hasattr(memory_facade, "add_memory"):
            return ReplayReport(still_deferred=len(entries), remaining=len(entries))

        now = time.time()
        committed = still_deferred = refused = expired = 0
        remaining: list[dict[str, Any]] = []

        for entry in entries:
            age = now - float(entry.get("queued_at") or now)
            attempts = int(entry.get("attempts") or 0)
            if age > MAX_AGE_S or attempts >= MAX_ATTEMPTS:
                expired += 1
                logger.info(
                    "Dropping a deferred memory write from %s after %.1fh and "
                    "%d attempt(s) — it was never accepted.",
                    entry.get("origin"), age / 3600.0, attempts,
                )
                continue

            entry["attempts"] = attempts + 1
            ok, reason = await self._attempt(memory_facade, entry)
            if ok:
                committed += 1
                continue
            if is_deferral(reason):
                still_deferred += 1
                entry["reason"] = str(reason)
                remaining.append(entry)
            else:
                refused += 1
                logger.info(
                    "Dropping a held write from %s: refused on the merits (%s).",
                    entry.get("origin"), reason,
                )

        try:
            await self._store_async(remaining)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "deferred_retention", exc, severity="warning",
                action="deferred-retention queue could not be rewritten after replay",
            )

        report = ReplayReport(
            committed=committed,
            still_deferred=still_deferred,
            refused=refused,
            expired=expired,
            remaining=len(remaining),
        )
        if committed or refused or expired:
            logger.info("Deferred-retention replay: %s.", report.narrative())
        return report

    async def _attempt(
        self, memory_facade: Any, entry: Mapping[str, Any]
    ) -> tuple[bool, str]:
        try:
            result = memory_facade.add_memory(
                entry["text"], metadata=dict(entry.get("metadata") or {})
            )
            if hasattr(result, "__await__"):
                result = await result
            if result:
                return True, ""
            status = getattr(memory_facade, "_last_add_memory_status", None)
            reason = ""
            if isinstance(status, Mapping):
                reason = str(status.get("reason") or "")
            return False, reason or str(entry.get("reason") or "unknown")
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation(
                "deferred_retention", exc, severity="debug",
                action="deferred memory write retry raised",
            )
            # An exception is a transient fault, not a verdict: keep holding.
            return False, "retry_failed_temporarily"


_QUEUE: DeferredRetentionQueue | None = None


def get_deferred_retention_queue() -> DeferredRetentionQueue:
    global _QUEUE
    if _QUEUE is None:
        _QUEUE = DeferredRetentionQueue()
    return _QUEUE


def reset_deferred_retention_queue() -> None:
    """Test seam: drop the process-wide handle so a new path is picked up."""

    global _QUEUE
    _QUEUE = None
