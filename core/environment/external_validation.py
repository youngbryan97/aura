"""External task proof gate for environment runs.

The environment kernel can run strict-real, simulated-canary, and fixture
adapters.  This gate keeps those modes honest: plumbing checks and placeholder
benchmarks are allowed as canaries, but they cannot be reported as external
capability proof unless they have a replayable trace, a live adapter, and a
closed run record from a non-placeholder mode.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Literal


ProofLevel = Literal["none", "fixture", "simulated", "strict_real"]


@dataclass(frozen=True)
class ExternalTaskEvidence:
    adapter_id: str
    mode: str
    proof_level: ProofLevel
    trace_rows: int
    closed_runs: int
    success_runs: int
    death_runs: int
    contaminated_runs: int
    placeholder_detected: bool
    reasons: tuple[str, ...] = ()
    receipts: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.reasons

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["passed"] = self.passed
        return payload


class ExternalTaskProofGate:
    """Verify that a run is usable as task evidence, not just architecture smoke."""

    PLACEHOLDER_TOKENS = (
        "placeholder",
        "stub",
        "always pass",
        "not real evaluation",
        "would override",
        "todo",
    )

    def evaluate_kernel(self, kernel: Any, *, require_strict_real: bool = False) -> ExternalTaskEvidence:
        adapter = getattr(kernel, "adapter", None)
        adapter_id = getattr(adapter, "environment_id", getattr(kernel, "environment_id", "unknown"))
        manager = getattr(kernel, "run_manager", None)
        mode = str(getattr(manager, "mode", "unknown"))
        proof_level = self._proof_level(mode)
        trace_rows = len(getattr(getattr(kernel, "blackbox", None), "rows", []) or [])
        records = list(getattr(manager, "records", []) or [])
        closed = [record for record in records if getattr(record, "ended_at", None)]
        success = [record for record in closed if getattr(record, "terminal_reason", "") == "success"]
        deaths = [record for record in closed if getattr(record, "terminal_reason", "") == "death"]
        contaminated = [record for record in closed if getattr(record, "contaminated", False)]
        reasons: list[str] = []

        placeholder = self._adapter_looks_placeholder(adapter)
        if require_strict_real and proof_level != "strict_real":
            reasons.append("strict_real_required")
        if proof_level == "none":
            reasons.append("unrecognized_or_unlabeled_mode")
        if placeholder:
            reasons.append("placeholder_adapter_or_benchmark")
        if trace_rows <= 0:
            reasons.append("missing_blackbox_trace_rows")
        if not closed:
            reasons.append("missing_closed_run_record")
        if contaminated:
            reasons.append("contaminated_run")
        if adapter is not None and callable(getattr(adapter, "is_alive", None)):
            try:
                adapter_alive_known = adapter.is_alive()
            except Exception as exc:
                adapter_alive_known = False
                reasons.append(f"adapter_liveness_error:{type(exc).__name__}")
        else:
            adapter_alive_known = False
            reasons.append("adapter_liveness_unavailable")

        return ExternalTaskEvidence(
            adapter_id=str(adapter_id),
            mode=mode,
            proof_level=proof_level,
            trace_rows=trace_rows,
            closed_runs=len(closed),
            success_runs=len(success),
            death_runs=len(deaths),
            contaminated_runs=len(contaminated),
            placeholder_detected=placeholder,
            reasons=tuple(reasons),
            receipts={
                "adapter_class": adapter.__class__.__name__ if adapter is not None else "",
                "records": [self._safe_record(record) for record in records[-5:]],
                "adapter_alive_observed": adapter_alive_known,
            },
        )

    def evaluate_trace_file(
        self,
        trace_path: str | Path,
        *,
        mode: str,
        adapter_id: str,
        require_strict_real: bool = False,
    ) -> ExternalTaskEvidence:
        path = Path(trace_path)
        reasons: list[str] = []
        trace_rows = 0
        if not path.exists():
            reasons.append("trace_file_missing")
        else:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        trace_rows += 1
                        try:
                            json.loads(line)
                        except json.JSONDecodeError:
                            reasons.append("trace_file_not_jsonl")
                            break
        proof_level = self._proof_level(mode)
        if require_strict_real and proof_level != "strict_real":
            reasons.append("strict_real_required")
        if proof_level == "none":
            reasons.append("unrecognized_or_unlabeled_mode")
        if trace_rows <= 0:
            reasons.append("missing_blackbox_trace_rows")

        return ExternalTaskEvidence(
            adapter_id=adapter_id,
            mode=mode,
            proof_level=proof_level,
            trace_rows=trace_rows,
            closed_runs=0,
            success_runs=0,
            death_runs=0,
            contaminated_runs=0,
            placeholder_detected=False,
            reasons=tuple(reasons),
            receipts={"trace_path": str(path)},
        )

    @staticmethod
    def _proof_level(mode: str) -> ProofLevel:
        normalized = str(mode).strip().lower()
        if normalized == "strict_real":
            return "strict_real"
        if normalized == "simulated_canary":
            return "simulated"
        if normalized == "fixture_replay":
            return "fixture"
        return "none"

    def _adapter_looks_placeholder(self, adapter: Any) -> bool:
        if adapter is None:
            return True
        flags = (
            getattr(adapter, "placeholder", False),
            getattr(adapter, "is_placeholder", False),
            getattr(adapter, "_placeholder", False),
        )
        if any(bool(flag) for flag in flags):
            return True
        doc = getattr(adapter.__class__, "__doc__", "") or ""
        name = adapter.__class__.__name__.lower()
        text = f"{name}\n{doc}".lower()
        return any(token in text for token in self.PLACEHOLDER_TOKENS)

    @staticmethod
    def _safe_record(record: Any) -> dict[str, Any]:
        if is_dataclass(record):
            payload = asdict(record)
        else:
            payload = dict(getattr(record, "__dict__", {}) or {})
        payload.pop("postmortem", None)
        return payload


__all__ = ["ExternalTaskEvidence", "ExternalTaskProofGate", "ProofLevel"]
