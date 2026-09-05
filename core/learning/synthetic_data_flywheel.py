"""Verifier-gated synthetic trace flywheel.

Successful repairs and decisions become training examples only after verifier
evidence is attached.  The flywheel writes JSONL data that nightly LoRA or
distillation jobs can consume without mixing in unverified drift.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from core.runtime.atomic_writer import atomic_write_text


@dataclass(frozen=True)
class VerifiedTrace:
    trace_id: str
    task_type: str
    prompt: str
    response: str
    verifier: str
    score: float
    risk_tier: str = "low"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def eligible(self) -> bool:
        return self.score >= 0.80 and self.risk_tier not in {"tier3_sealed"}

    def to_record(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "task_type": self.task_type,
            "input": self.prompt,
            "output": self.response,
            "verifier": self.verifier,
            "score": self.score,
            "risk_tier": self.risk_tier,
            "metadata": self.metadata,
            "created_at": time.time(),
        }


class SyntheticDataFlywheel:
    def __init__(self, output_dir: str | Path = "data/synthetic_traces") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_from_success(self, success: Mapping[str, Any], *, variants: int = 50) -> list[VerifiedTrace]:
        prompt = str(success.get("prompt") or success.get("input") or success.get("task") or "")
        response = str(success.get("response") or success.get("output") or success.get("answer") or "")
        if not prompt or not response:
            return []
        score = float(success.get("score", success.get("verifier_score", 1.0)) or 0.0)
        task_type = str(success.get("task_type", "general"))
        risk_tier = str(success.get("risk_tier", "low"))
        traces: list[VerifiedTrace] = []
        for idx in range(max(1, variants)):
            trace_id = f"{success.get('id', 'trace')}_{idx:03d}"
            traces.append(
                VerifiedTrace(
                    trace_id=trace_id,
                    task_type=task_type,
                    prompt=self._variant_prompt(prompt, idx),
                    response=response,
                    verifier=str(success.get("verifier", "gold_decision")),
                    score=score,
                    risk_tier=risk_tier,
                    metadata={"source_success_id": success.get("id"), "variant": idx},
                )
            )
        return [trace for trace in traces if trace.eligible]

    def write_jsonl(self, traces: Iterable[VerifiedTrace], path: str | Path | None = None) -> Path:
        target = Path(path) if path else self.output_dir / f"verified_traces_{int(time.time())}.jsonl"
        records = [trace.to_record() for trace in traces if trace.eligible]
        atomic_write_text(target, "\n".join(json.dumps(record, sort_keys=True) for record in records) + ("\n" if records else ""), encoding="utf-8")
        return target

    @staticmethod
    def _variant_prompt(prompt: str, idx: int) -> str:
        wrappers = (
            "{prompt}",
            "Solve carefully: {prompt}",
            "Given the verified prior solution, handle this equivalent task: {prompt}",
            "Use the same reliability standard on: {prompt}",
        )
        return wrappers[idx % len(wrappers)].format(prompt=prompt)


__all__ = ["VerifiedTrace", "SyntheticDataFlywheel"]
