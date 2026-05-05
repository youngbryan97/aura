"""Ledger for recording outcome consequences across runs."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class OutcomeRecord:
    action_name: str
    environment_id: str
    context_id: str
    successes: int = 0
    failures: int = 0
    average_success_score: float = 0.0
    semantic_consequences: list[str] = field(default_factory=list)


class OutcomeLedger:
    """Persists aggregated success/failure rates of actions in contexts."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.records: dict[str, OutcomeRecord] = {}
        if self.path and self.path.exists():
            self.load()

    def record_outcome(self, action: str, env_id: str, ctx_id: str, success: bool, score: float, consequences: list[str]) -> None:
        key = f"{env_id}::{ctx_id}::{action}"
        if key not in self.records:
            self.records[key] = OutcomeRecord(action_name=action, environment_id=env_id, context_id=ctx_id)
        
        rec = self.records[key]
        if success:
            rec.successes += 1
        else:
            rec.failures += 1
            
        total = rec.successes + rec.failures
        rec.average_success_score = ((rec.average_success_score * (total - 1)) + score) / total
        
        # Keep unique consequences
        for c in consequences:
            if c not in rec.semantic_consequences:
                rec.semantic_consequences.append(c)

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({k: asdict(v) for k, v in self.records.items()}, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def load(self) -> None:
        if not self.path or not self.path.exists():
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.records = {k: OutcomeRecord(**v) for k, v in data.items()}

__all__ = ["OutcomeLedger", "OutcomeRecord"]
