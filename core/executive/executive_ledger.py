from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict


class ExecutiveLedger:
    """Append-only executive decision log.

    This is intentionally simple: a JSONL file that records what was proposed,
    what the executive decided, and when approved intents completed.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: Dict[str, Any]) -> None:
        payload = dict(event)
        payload.setdefault("timestamp", time.time())
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
