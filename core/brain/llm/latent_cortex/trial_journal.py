"""Durable, resumable trial receipts for long experiment runs.

CP126 ``51654706``. Experiment runners accumulated results in
process-local dicts and returned them only after every callback had
finished. There was no run id, no durable per-trial record, no index of
completed work, no resume validation, and an exception anywhere lost the
whole run. A crash after hours discarded every completed trial, and a
restart re-ran them — which for order-sensitive state is not merely
wasteful but can produce different numbers the second time.

A journal makes the run's progress a fact on disk rather than a value in
memory:

* every trial is appended immediately, as an immutable line;
* the journal is bound to a MANIFEST digest describing the run, so a resume
  can only ever attach to the same experiment — resuming a journal from a
  different task set or arm list is refused rather than silently mixed;
* a failing trial is recorded as an explicit failure receipt instead of
  propagating and destroying the completed work beside it;
* completed trials are indexed by key, so a resumed run skips exactly what
  it already did and nothing else.

Append is atomic in the way that matters here: one line per trial, opened
in append mode and flushed, so a crash mid-write truncates at most the
final partial line, which parsing skips. That is the standard durability
contract for an append-only journal and it is honest about its limit — it
does not claim fsync-per-trial durability against power loss.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def manifest_digest(manifest: Any) -> str:
    """Stable digest of the run manifest a journal is bound to."""
    return hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
    ).hexdigest()


@dataclass
class TrialRecord:
    """One trial's immutable outcome."""

    key: str
    ok: bool
    payload: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    at_unix: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "ok": self.ok,
            "payload": dict(self.payload),
            "error": self.error,
            "at_unix": round(self.at_unix, 3),
        }

    @classmethod
    def from_dict(cls, data: Any) -> TrialRecord | None:
        if not isinstance(data, dict):
            return None
        key = str(data.get("key") or "")
        if not key:
            return None
        payload = data.get("payload")
        return cls(
            key=key,
            ok=bool(data.get("ok")),
            payload=dict(payload) if isinstance(payload, dict) else {},
            error=str(data.get("error") or ""),
            at_unix=float(data.get("at_unix") or 0.0),
        )


class TrialJournal:
    """Append-only journal for one experiment run.

    ``run_id`` identifies the run; ``manifest`` describes what is being run
    and is digested so a resume cannot attach to a different experiment.
    """

    HEADER_KIND = "aura.trial_journal.v1"

    def __init__(
        self,
        path: str | Path,
        *,
        manifest: Any,
        run_id: str = "",
    ) -> None:
        self.path = Path(str(path))
        self.manifest_sha256 = manifest_digest(manifest)
        self.run_id = str(run_id or uuid.uuid4().hex)
        self._completed: dict[str, TrialRecord] = {}
        self._resumed = False
        self._skipped_corrupt = 0

    # ── lifecycle ───────────────────────────────────────────────────────
    def open(self) -> TrialJournal:
        """Create or resume the journal, refusing a mismatched manifest."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self._resume()
        else:
            self._write_header()
        return self

    def _write_header(self) -> None:
        header = {
            "kind": self.HEADER_KIND,
            "run_id": self.run_id,
            "manifest_sha256": self.manifest_sha256,
            "started_at_unix": round(time.time(), 3),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(header, sort_keys=True) + "\n")
            handle.flush()

    def _resume(self) -> None:
        header: dict[str, Any] | None = None
        for entry in self._read_lines():
            if header is None:
                if entry.get("kind") != self.HEADER_KIND:
                    raise ValueError(
                        f"trial_journal_missing_header:{self.path}"
                    )
                header = entry
                continue
            record = TrialRecord.from_dict(entry)
            if record is not None:
                self._completed[record.key] = record
        if header is None:
            # An empty or wholly unreadable file is not a resumable run.
            self._write_header()
            return
        existing_digest = str(header.get("manifest_sha256") or "")
        if existing_digest != self.manifest_sha256:
            raise ValueError(
                "trial_journal_manifest_mismatch:"
                f"{existing_digest[:12]}!={self.manifest_sha256[:12]}"
            )
        self.run_id = str(header.get("run_id") or self.run_id)
        self._resumed = True

    def _read_lines(self) -> Iterator[dict[str, Any]]:
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except ValueError:
                    # A truncated final line is the expected shape of a
                    # crash mid-append; skip it rather than lose the run.
                    self._skipped_corrupt += 1
                    continue
                if isinstance(entry, dict):
                    yield entry

    # ── use ─────────────────────────────────────────────────────────────
    @property
    def resumed(self) -> bool:
        return self._resumed

    @property
    def skipped_corrupt_lines(self) -> int:
        return self._skipped_corrupt

    def completed_keys(self) -> set[str]:
        return set(self._completed)

    def is_complete(self, key: str) -> bool:
        return key in self._completed

    def get(self, key: str) -> TrialRecord | None:
        return self._completed.get(key)

    def records(self) -> list[TrialRecord]:
        return list(self._completed.values())

    def record(
        self,
        key: str,
        *,
        ok: bool,
        payload: dict[str, Any] | None = None,
        error: str = "",
    ) -> TrialRecord:
        """Append one trial outcome and remember it."""
        entry = TrialRecord(key=str(key), ok=bool(ok), payload=dict(payload or {}), error=str(error))
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry.to_dict(), sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._completed[entry.key] = entry
        return entry

    def run_trial(self, key: str, work: Any) -> TrialRecord:
        """Execute ``work()`` once, journaling success OR failure.

        A trial that raises becomes a recorded FAILURE rather than an
        exception that discards every completed trial beside it. The
        exception type and message are kept so the failure is diagnosable,
        and the run can decide whether to continue.
        """
        existing = self._completed.get(str(key))
        if existing is not None:
            return existing
        try:
            payload = work()
        except (
            ArithmeticError, AttributeError, BufferError, EOFError, LookupError,
            MemoryError, OSError, RuntimeError, TypeError, ValueError,
        ) as exc:
            return self.record(
                key, ok=False, error=f"{type(exc).__name__}: {exc}"[:400],
            )
        return self.record(
            key,
            ok=True,
            payload=payload if isinstance(payload, dict) else {"value": payload},
        )

    def summary(self) -> dict[str, Any]:
        succeeded = sum(1 for record in self._completed.values() if record.ok)
        failed = len(self._completed) - succeeded
        return {
            "run_id": self.run_id,
            "manifest_sha256": self.manifest_sha256,
            "path": str(self.path),
            "resumed": self._resumed,
            "trials": len(self._completed),
            "succeeded": succeeded,
            "failed": failed,
            "skipped_corrupt_lines": self._skipped_corrupt,
        }


__all__ = ["TrialJournal", "TrialRecord", "manifest_digest"]
