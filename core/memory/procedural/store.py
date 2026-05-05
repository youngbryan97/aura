"""In-memory procedural store with JSON persistence hooks."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .schema import ProcedureRecord


class ProceduralMemoryStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.records: dict[str, ProcedureRecord] = {}
        if self.path and self.path.exists():
            self.load()

    def upsert(self, record: ProcedureRecord) -> ProcedureRecord:
        existing = self.records.get(record.procedure_id)
        if existing:
            existing.success_count += record.success_count
            existing.failure_count += record.failure_count
            existing.mean_outcome_score = (
                existing.mean_outcome_score + record.mean_outcome_score
            ) / 2.0
            existing.risk_score = max(existing.risk_score, record.risk_score)
            existing.examples = (existing.examples + record.examples)[-20:]
            return existing
        self.records[record.procedure_id] = record
        return record

    def record(self, environment_family: str, context_signature: str, procedure: dict) -> None:
        import hashlib
        action = procedure.get("action", "unknown")
        pid = f"{environment_family}_{context_signature}_{action}"
        pid_hash = hashlib.md5(pid.encode()).hexdigest()[:8]
        
        record = ProcedureRecord(
            procedure_id=f"proc_{pid_hash}",
            environment_family=environment_family,
            context_signature=context_signature,
            option_name=action,
            parameters=procedure.get("parameters", {}),
            success_count=1,
            failure_count=0,
            mean_outcome_score=1.0,
            risk_score=0.1,
            success_conditions=procedure.get("effect", []),
            confidence=0.5
        )
        self.upsert(record)

    def retrieve(self, *, environment_family: str, context_signature: str, goal: str = "") -> list[ProcedureRecord]:
        matches = [
            rec
            for rec in self.records.values()
            if rec.environment_family == environment_family
            and (rec.context_signature == context_signature or context_signature in rec.context_signature or rec.context_signature in context_signature)
            and (not goal or goal.lower() in rec.option_name.lower() or goal.lower() in " ".join(rec.success_conditions).lower())
        ]
        return sorted(matches, key=lambda rec: (rec.confidence, rec.success_count, -rec.risk_score), reverse=True)

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps([asdict(rec) for rec in self.records.values()], indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def load(self) -> None:
        if not self.path or not self.path.exists():
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.records = {raw["procedure_id"]: ProcedureRecord(**raw) for raw in data}


__all__ = ["ProceduralMemoryStore"]
