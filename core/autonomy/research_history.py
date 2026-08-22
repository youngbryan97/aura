"""The durable record of what research actually happened.

Records were appended straight to a JSONL file with no write gateway, no
lock, no chain hash and no schema envelope, so nothing could tell a
tampered or truncated history from a genuine one and two processes could
interleave lines (CP126 ``9e6e006e``). And reload restored only the
records and the cycle count, so the cooldown clock and every per-goal
failure count reset — a restart could immediately rerun research it had
just finished, or retry a goal that had already spent its budget
(``a9ed8b95``).

Lifted out of ``ResearchCycle`` because persistence is not the cycle's
job and the class was well past the size the ratchet allows.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from core.governance_context import local_internal_governed_scope
from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.lockdep import checked_lock

__all__ = ["HISTORY_SCHEMA", "ResearchHistory"]

HISTORY_SCHEMA = "aura.autonomy.research_record.v2"


class ResearchHistory:
    """Append-only, chained, gateway-written history for one cycle."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.chain_head: str = ""
        self._lock = checked_lock("core.autonomy.research_history")

    def append(self, payload: dict[str, Any]) -> str:
        """Write one record chained to the last. Returns its digest."""
        with self._lock:
            envelope = {
                "schema": HISTORY_SCHEMA,
                "previous_sha256": self.chain_head,
                "record": payload,
            }
            line = json.dumps(envelope, sort_keys=True, separators=(",", ":"))
            envelope["record_sha256"] = hashlib.sha256(
                line.encode("utf-8")
            ).hexdigest()
            with local_internal_governed_scope(
                "autonomy.research_cycle.history",
                domain="memory_write",
                receipt_prefix="research-history-append",
                constraints={
                    "artifact": HISTORY_SCHEMA,
                    "operation": "append_only",
                    "record_sha256": envelope["record_sha256"],
                },
            ):
                get_file_write_gateway().append_text(
                    self.path,
                    json.dumps(envelope, sort_keys=True, separators=(",", ":"))
                    + "\n",
                    source="autonomy.research_cycle.history",
                )
            self.chain_head = envelope["record_sha256"]
        return envelope["record_sha256"]

    def reset_reader(self) -> None:
        """Reset chain verification before replaying this history from disk."""
        with self._lock:
            self.chain_head = ""

    def read_payload(self, line: str) -> dict[str, Any] | None:
        """One record from a stored line, or None when it does not verify.

        Accepts the pre-envelope shape too, so an existing history still
        loads; a row that CLAIMS a digest and does not match it is refused,
        because that is the only case where refusing tells you something.
        """
        data = json.loads(line)
        if not isinstance(data, dict):
            return None
        if "record" not in data:
            return data
        expected = data.pop("record_sha256", "")
        recomputed = hashlib.sha256(
            json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        previous = data.get("previous_sha256")
        if (
            not expected
            or expected != recomputed
            or not isinstance(previous, str)
            or previous != self.chain_head
        ):
            return None
        self.chain_head = expected
        record = data.get("record")
        return record if isinstance(record, dict) else None
