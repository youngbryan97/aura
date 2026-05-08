"""In-memory procedural store with JSON persistence hooks."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from core.runtime.atomic_writer import AtomicWriteError, atomic_write_json, read_json_envelope

from .schema import ProcedureRecord

PROCEDURAL_STORE_SCHEMA_VERSION = 1


class ProceduralMemoryStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.records: dict[str, ProcedureRecord] = {}
        if self.path and self.path.exists():
            self.load()

    def upsert(self, record: ProcedureRecord) -> ProcedureRecord:
        existing = self.records.get(record.procedure_id)
        if existing:
            existing_total = max(1, existing.success_count + existing.failure_count)
            record_total = max(1, record.success_count + record.failure_count)
            existing.success_count += record.success_count
            existing.failure_count += record.failure_count
            existing.mean_outcome_score = (
                (existing.mean_outcome_score * existing_total)
                + (record.mean_outcome_score * record_total)
            ) / (existing_total + record_total)
            existing.risk_score = max(existing.risk_score, record.risk_score)
            existing.preconditions = list(dict.fromkeys(existing.preconditions + record.preconditions))[-20:]
            existing.success_conditions = list(dict.fromkeys(existing.success_conditions + record.success_conditions))[-20:]
            existing.failure_conditions = list(dict.fromkeys(existing.failure_conditions + record.failure_conditions))[-20:]
            existing.examples = (existing.examples + record.examples)[-20:]
            return existing
        self.records[record.procedure_id] = record
        return record

    def record(self, environment_family: str, context_signature: str, procedure: dict[str, Any]) -> None:
        action = procedure.get("action", "unknown")
        pid = f"{environment_family}_{context_signature}_{action}"
        pid_hash = hashlib.sha256(pid.encode("utf-8")).hexdigest()[:12]
        success = bool(procedure.get("success", True))
        outcome_score = float(procedure.get("outcome_score", 1.0 if success else 0.0) or 0.0)
        risk_score = float(procedure.get("risk_score", 0.1) or 0.0)
        
        record = ProcedureRecord(
            procedure_id=f"proc_{pid_hash}",
            environment_family=environment_family,
            context_signature=context_signature,
            option_name=action,
            preconditions=list(procedure.get("preconditions", [])),
            action_sequence_template=[
                action,
                *[f"{key}={value}" for key, value in sorted(dict(procedure.get("parameters", {}) or {}).items())],
            ],
            success_conditions=list(procedure.get("effect", [])),
            failure_conditions=list(procedure.get("failure_conditions", [])),
            success_count=1 if success else 0,
            failure_count=0 if success else 1,
            mean_outcome_score=max(0.0, min(1.0, outcome_score)),
            risk_score=max(0.0, min(1.0, risk_score)),
            examples=[json.dumps(procedure, sort_keys=True, default=str)[:500]],
        )
        self.upsert(record)

    def record_outcome(
        self,
        *,
        environment_family: str,
        context_signature: str,
        action: str,
        parameters: dict[str, Any] | None = None,
        observed_events: list[str] | None = None,
        success: bool,
        outcome_score: float,
        risk_score: float = 0.0,
        failure_conditions: list[str] | None = None,
    ) -> None:
        self.record(
            environment_family,
            context_signature,
            {
                "action": action,
                "parameters": parameters or {},
                "effect": observed_events or [],
                "success": success,
                "outcome_score": outcome_score,
                "risk_score": risk_score,
                "failure_conditions": failure_conditions or [],
            },
        )

    def retrieve(self, *, environment_family: str, context_signature: str, goal: str = "") -> list[ProcedureRecord]:
        matches = [
            rec
            for rec in self.records.values()
            if rec.environment_family in {environment_family, "general"}
            and (rec.context_signature == context_signature or context_signature in rec.context_signature or rec.context_signature in context_signature)
            and (not goal or goal.lower() in rec.option_name.lower() or goal.lower() in " ".join(rec.success_conditions).lower())
        ]
        return sorted(matches, key=lambda rec: (rec.confidence, rec.success_count, -rec.risk_score), reverse=True)

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            self.path,
            [asdict(rec) for rec in self.records.values()],
            schema_name="procedural_memory_store",
            schema_version=PROCEDURAL_STORE_SCHEMA_VERSION,
        )

    def load(self) -> None:
        if not self.path or not self.path.exists():
            return
        try:
            envelope = read_json_envelope(self.path)
            data = envelope.get("payload", [])
        except AtomicWriteError:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("procedural memory payload must be a list")
        self.records = {raw["procedure_id"]: ProcedureRecord(**raw) for raw in data}


__all__ = ["ProceduralMemoryStore"]
