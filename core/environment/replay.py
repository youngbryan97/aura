"""Replay black-box traces without a live environment."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ReplayResult:
    ok: bool
    rows: list[dict[str, Any]] = field(default_factory=list)
    corrupt_rows: list[int] = field(default_factory=list)
    postmortem: dict[str, Any] = field(default_factory=dict)


class EnvironmentTraceReplay:
    def load(self, path: str | Path) -> ReplayResult:
        rows: list[dict[str, Any]] = []
        corrupt: list[int] = []
        previous = ""
        for idx, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                expected_prev = row.get("previous_hash", "")
                if expected_prev != previous:
                    corrupt.append(idx)
                row_hash = row.get("row_hash", "")
                payload = dict(row)
                payload["row_hash"] = ""
                actual = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()
                if row_hash != actual:
                    corrupt.append(idx)
                previous = row_hash
                rows.append(row)
            except Exception:
                corrupt.append(idx)
        postmortem = {}
        if rows:
            last = rows[-1]
            postmortem = {
                "terminal_event": last.get("execution_result", {}).get("error") or last.get("outcome_assessment", {}).get("lesson", ""),
                "final_sequence_id": last.get("sequence_id"),
                "trace_rows": len(rows),
                "corrupt_rows": corrupt,
            }
        return ReplayResult(ok=not corrupt, rows=rows, corrupt_rows=corrupt, postmortem=postmortem)


__all__ = ["ReplayResult", "EnvironmentTraceReplay"]
