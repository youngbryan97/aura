"""Observation records for environment adapters."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except Exception:
        if isinstance(value, dict):
            return {str(k): _json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_json_safe(v) for v in value]
        return repr(value)


@dataclass
class Observation:
    environment_id: str
    run_id: str
    sequence_id: int
    timestamp: float = field(default_factory=time.time)
    context_id: str | None = None
    raw: Any = None
    text: str | None = None
    image_ref: str | None = None
    structured: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def stable_hash(self) -> str:
        payload = {
            "environment_id": self.environment_id,
            "run_id": self.run_id,
            "sequence_id": self.sequence_id,
            "context_id": self.context_id,
            "text": self.text,
            "structured": _json_safe(self.structured),
            "metadata": _json_safe(self.metadata),
            "raw": _json_safe(self.raw),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def to_json_safe(self) -> dict[str, Any]:
        data = asdict(self)
        data["raw"] = _json_safe(self.raw)
        return _json_safe(data)


__all__ = ["Observation", "_json_safe"]
