"""Black-box telemetry for every environment step."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .observation import _json_safe


@dataclass
class BlackBoxRow:
    run_id: str
    sequence_id: int
    environment_id: str
    context_id: str | None
    raw_observation_hash: str
    parsed_state_ref: str
    belief_hash_before: str
    attention_claims: list[dict[str, Any]] = field(default_factory=list)
    selected_goal: str = ""
    selected_option: str = ""
    action_intent: dict[str, Any] = field(default_factory=dict)
    simulation_result: dict[str, Any] = field(default_factory=dict)
    gateway_decision: dict[str, Any] = field(default_factory=dict)
    will_receipt_id: str | None = None
    authority_receipt_id: str | None = None
    command_spec: dict[str, Any] = field(default_factory=dict)
    execution_result: dict[str, Any] = field(default_factory=dict)
    semantic_events: list[dict[str, Any]] = field(default_factory=list)
    outcome_assessment: dict[str, Any] = field(default_factory=dict)
    belief_hash_after: str = ""
    learning_updates: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0
    previous_hash: str = ""
    row_hash: str = ""
    timestamp: float = field(default_factory=time.time)

    def compute_hash(self) -> str:
        payload = asdict(self)
        payload["row_hash"] = ""
        return hashlib.sha256(json.dumps(_json_safe(payload), sort_keys=True).encode("utf-8")).hexdigest()


class BlackBoxRecorder:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.rows: list[BlackBoxRow] = []
        self._last_hash = ""
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, row: BlackBoxRow) -> BlackBoxRow:
        row.previous_hash = self._last_hash
        row.row_hash = row.compute_hash()
        self._last_hash = row.row_hash
        self.rows.append(row)
        if self.path:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(_json_safe(asdict(row)), sort_keys=True) + "\n")
        return row


__all__ = ["BlackBoxRow", "BlackBoxRecorder"]
