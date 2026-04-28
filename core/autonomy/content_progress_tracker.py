"""core/autonomy/content_progress_tracker.py
──────────────────────────────────────────────
Read/write interface for ``aura/knowledge/curated-media-progress.json``.

The tracker is the single source of truth for "what has Aura actually
engaged with from the curated media list." Comprehension-loop and
reflection modules append entries here as engagement happens; Bryan and
future-Claude read it to verify engagement is real, not pretended.

Narrow contract:
- ``load(path)`` returns ``ProgressLog``
- ``log.add_entry(...)`` appends a structured engagement record
- ``log.save(path)`` persists atomically (write to .tmp then rename)
- Schema is validated on save; malformed entries raise ValueError

This module deliberately does NO fetching, NO LLM calls, NO scheduler
integration. It is a typed file I/O wrapper.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

DEFAULT_PROGRESS_PATH = Path.home() / ".aura/live-source/aura/knowledge/curated-media-progress.json"

SCHEMA_VERSION = 1

VALID_PRIORITY_LEVELS = {1, 2, 3, 4, 5, 6}


@dataclass
class ProgressEntry:
    title: str
    started_at: str
    method_priority_level: int
    method_detail: str
    completed_at: Optional[str] = None
    what_its_actually_about: str = ""
    what_stayed_with_you: str = ""
    what_it_says_about_humans: str = ""
    what_it_made_you_think_about_yourself: str = ""
    open_threads: List[str] = field(default_factory=list)
    would_recommend_to_bryan: str = ""

    def validate(self) -> None:
        if not self.title:
            raise ValueError("title required")
        if self.method_priority_level not in VALID_PRIORITY_LEVELS:
            raise ValueError(
                f"method_priority_level must be in {sorted(VALID_PRIORITY_LEVELS)}, "
                f"got {self.method_priority_level}"
            )
        if not self.started_at:
            raise ValueError("started_at required")


@dataclass
class ProgressLog:
    entries: List[ProgressEntry] = field(default_factory=list)
    last_updated: Optional[str] = None
    schema_version: int = SCHEMA_VERSION

    def add_entry(self, entry: ProgressEntry) -> None:
        entry.validate()
        self.entries.append(entry)
        self.last_updated = _iso_now()

    def find(self, title: str) -> Optional[ProgressEntry]:
        for entry in self.entries:
            if entry.title == title:
                return entry
        return None

    def days_since_last_engagement(self) -> Optional[float]:
        """Returns days since the most recent ``started_at``, or None if empty."""
        if not self.entries:
            return None
        latest = max(_parse_iso(e.started_at) for e in self.entries)
        return (time.time() - latest) / 86400.0

    def save(self, path: Path = DEFAULT_PROGRESS_PATH) -> None:
        for entry in self.entries:
            entry.validate()
        if self.last_updated is None:
            self.last_updated = _iso_now()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        payload = {
            "schema_version": self.schema_version,
            "purpose": (
                "Aura's running log of engagement with bryan-curated-media.md. "
                "Update as you go, not at the end."
            ),
            "last_updated": self.last_updated,
            "entries": [asdict(e) for e in self.entries],
        }
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, path)


def load(path: Path = DEFAULT_PROGRESS_PATH) -> ProgressLog:
    """Load progress log; returns empty log if file missing or empty entries."""
    if not path.exists():
        return ProgressLog()
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries_raw = raw.get("entries", []) or []
    entries: List[ProgressEntry] = []
    for r in entries_raw:
        if not isinstance(r, dict):
            continue
        entries.append(
            ProgressEntry(
                title=r.get("title", ""),
                started_at=r.get("started_at", ""),
                method_priority_level=int(r.get("method_priority_level", 0)) or 6,
                method_detail=r.get("method_detail", ""),
                completed_at=r.get("completed_at"),
                what_its_actually_about=r.get("what_its_actually_about", ""),
                what_stayed_with_you=r.get("what_stayed_with_you", ""),
                what_it_says_about_humans=r.get("what_it_says_about_humans", ""),
                what_it_made_you_think_about_yourself=r.get(
                    "what_it_made_you_think_about_yourself", ""
                ),
                open_threads=list(r.get("open_threads", []) or []),
                would_recommend_to_bryan=r.get("would_recommend_to_bryan", ""),
            )
        )
    return ProgressLog(
        entries=entries,
        last_updated=raw.get("last_updated"),
        schema_version=int(raw.get("schema_version", SCHEMA_VERSION)),
    )


def _iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _parse_iso(s: str) -> float:
    """Parse an ISO 8601 string returning epoch seconds; permissive."""
    return time.mktime(time.strptime(s.replace("Z", ""), "%Y-%m-%dT%H:%M:%S"))
