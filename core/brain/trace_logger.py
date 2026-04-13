# core/brain/trace_logger.py
import atexit
import json
import logging
import time
from pathlib import Path
from typing import Any


class TraceLogger:
    def __init__(self, path: str | Path = "~/.aura/traces/decisions.jsonl"):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")
        atexit.register(self.close)

    def log(self, record: dict[str, Any]) -> None:
        rec = {
            "ts": time.time(),
            **record
        }
        self._fh.write(json.dumps(rec, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception as _e:
            logging.debug('Ignored Exception in trace_logger.py: %s', _e)

    def __del__(self) -> None:
        self.close()
