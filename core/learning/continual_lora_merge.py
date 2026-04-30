"""Continual LoRA merge/distillation planner.

The heavy training run remains an explicit scheduled job, but this module makes
the merge cycle concrete: collect verified trace shards, build the mlx-lm
command, write provenance, and refuse promotion without validation evidence.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from core.runtime.atomic_writer import atomic_write_text


@dataclass(frozen=True)
class MergeDistillPlan:
    model_path: str
    adapter_inputs: tuple[str, ...]
    data_files: tuple[str, ...]
    output_adapter: str
    train_command: tuple[str, ...]
    validation_required: tuple[str, ...] = ("identity_validation", "behavioral_contracts", "hidden_eval")
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_path": self.model_path,
            "adapter_inputs": list(self.adapter_inputs),
            "data_files": list(self.data_files),
            "output_adapter": self.output_adapter,
            "train_command": list(self.train_command),
            "validation_required": list(self.validation_required),
            "created_at": self.created_at,
        }


class ContinualLoRAMerger:
    def __init__(self, model_path: str, work_dir: str | Path = "data/lora_continual") -> None:
        self.model_path = model_path
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def plan(
        self,
        *,
        adapters: Iterable[str | Path],
        data_files: Iterable[str | Path],
        output_adapter: str | Path | None = None,
        iters: int = 120,
        learning_rate: str = "8e-6",
    ) -> MergeDistillPlan:
        adapter_list = tuple(str(Path(a)) for a in adapters)
        data_list = tuple(str(Path(d)) for d in data_files)
        out = str(Path(output_adapter) if output_adapter else self.work_dir / f"adapter_{int(time.time())}")
        merged_data = self.work_dir / f"distill_manifest_{int(time.time())}.json"
        atomic_write_text(
            merged_data,
            json.dumps({"adapters": adapter_list, "data_files": data_list, "created_at": time.time()}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        command = (
            "python",
            "-m",
            "mlx_lm.lora",
            "--model",
            self.model_path,
            "--data",
            data_list[0] if data_list else str(merged_data),
            "--iters",
            str(iters),
            "--learning-rate",
            learning_rate,
            "--adapter-path",
            out,
        )
        plan = MergeDistillPlan(self.model_path, adapter_list, data_list, out, command)
        atomic_write_text(self.work_dir / "latest_plan.json", json.dumps(plan.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return plan

    def promotion_allowed(self, validation: dict[str, Any]) -> bool:
        return (
            bool(validation.get("identity_validation", False))
            and bool(validation.get("behavioral_contracts", False))
            and bool(validation.get("hidden_eval", False))
            and float(validation.get("quality_delta", 0.0)) >= -0.03
        )


__all__ = ["MergeDistillPlan", "ContinualLoRAMerger"]
