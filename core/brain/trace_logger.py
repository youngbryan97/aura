# core/brain/trace_logger.py
import json
import logging
import time
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway


class TraceLogger:
    def __init__(self, path: str | Path = "~/.aura/traces/decisions.jsonl"):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: dict[str, Any]) -> None:
        rec = {
            "ts": time.time(),
            **record
        }
        get_file_write_gateway().append_text(
            self.path,
            json.dumps(rec, default=str) + "\n",
            encoding="utf-8",
            source="trace_logger.log",
        )

    def close(self) -> None:
        return None

    def __del__(self) -> None:
        self.close()
