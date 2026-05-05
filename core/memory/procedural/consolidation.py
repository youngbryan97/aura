"""Convert valid outcomes into procedural records."""
from __future__ import annotations

import hashlib

from .schema import ProcedureRecord


def procedure_from_outcome(
    *,
    option_name: str,
    environment_family: str,
    context_signature: str,
    actions: list[str],
    success: bool,
    outcome_score: float,
    risk_score: float,
    trace_id: str,
) -> ProcedureRecord:
    if not trace_id:
        raise ValueError("procedure_requires_valid_trace")
    pid = "proc_" + hashlib.sha256(
        f"{environment_family}:{option_name}:{context_signature}:{actions}".encode("utf-8")
    ).hexdigest()[:16]
    return ProcedureRecord(
        procedure_id=pid,
        option_name=option_name,
        environment_family=environment_family,
        context_signature=context_signature,
        preconditions=[context_signature],
        action_sequence_template=list(actions),
        success_conditions=["outcome_score_positive"],
        failure_conditions=["outcome_score_negative", "trace_invalid"],
        success_count=1 if success else 0,
        failure_count=0 if success else 1,
        mean_outcome_score=outcome_score,
        risk_score=risk_score,
        examples=[trace_id],
    )


__all__ = ["procedure_from_outcome"]
