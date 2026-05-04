"""Trace logging for embodied cognition loops."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class EmbodiedTraceRecord:
    timestamp: float
    domain: str
    observation_id: str
    context_id: str
    risk_level: str
    risk_score: float
    goal: str
    skill: str
    messages: List[str] = field(default_factory=list)
    action: Optional[str] = None
    action_decision: Optional[Dict[str, Any]] = None
    belief_uncertainty: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class EmbodiedTraceLogger:
    """In-memory plus optional JSONL audit trail."""

    def __init__(
        self,
        path: Optional[Path] = None,
        *,
        max_records: int = 2000,
        mirror_existing_trace: bool = True,
    ) -> None:
        self.path = Path(path) if path else None
        self.max_records = int(max_records)
        self.records: List[EmbodiedTraceRecord] = []
        self._existing_trace = None
        if mirror_existing_trace:
            try:
                from core.brain.trace_logger import TraceLogger

                self._existing_trace = TraceLogger("~/.aura/traces/embodied_cognition.jsonl")
            except Exception:
                self._existing_trace = None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, record: EmbodiedTraceRecord) -> None:
        self.records.append(record)
        self.records = self.records[-self.max_records :]
        payload = asdict(record)
        if self.path:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
        if self._existing_trace is not None:
            try:
                self._existing_trace.log({"type": "embodied_cognition", **payload})
            except Exception:
                pass

    def latest(self) -> Optional[EmbodiedTraceRecord]:
        return self.records[-1] if self.records else None

    def record_observation(
        self,
        *,
        domain: str,
        observation_id: str,
        context_id: str,
        risk_level: str,
        risk_score: float,
        goal: str,
        skill: str,
        messages: List[str],
        belief_uncertainty: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EmbodiedTraceRecord:
        record = EmbodiedTraceRecord(
            timestamp=time.time(),
            domain=domain,
            observation_id=observation_id,
            context_id=context_id,
            risk_level=risk_level,
            risk_score=risk_score,
            goal=goal,
            skill=skill,
            messages=list(messages[-5:]),
            belief_uncertainty=belief_uncertainty,
            metadata=dict(metadata or {}),
        )
        self.record(record)
        return record
