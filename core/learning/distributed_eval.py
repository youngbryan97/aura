"""Bounded local distributed evaluation.

This module gives Aura a real distributed evaluation primitive without cloud
provisioning or network replication. Work is spread across local worker
processes under explicit CPU and memory-aware caps, which is enough for
parallel successor/architecture experiments while LoRA or model jobs are live.
"""
from __future__ import annotations

import concurrent.futures
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, List, Optional, Sequence


@dataclass(frozen=True)
class DistributedEvalConfig:
    requested_workers: int = 2
    max_workers: int = 4
    min_free_mem_mb_per_worker: int = 512
    preserve_one_core: bool = True


@dataclass(frozen=True)
class DistributedEvalResult:
    worker_count: int
    input_count: int
    outputs: List[Any]
    runtime_s: float
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors and len(self.outputs) == self.input_count

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _available_memory_mb() -> Optional[float]:
    try:
        import psutil  # type: ignore

        return float(psutil.virtual_memory().available / (1024 * 1024))
    except Exception:
        return None


def safe_worker_count(config: DistributedEvalConfig) -> int:
    cpus = os.cpu_count() or 1
    cpu_cap = max(1, cpus - 1) if config.preserve_one_core else max(1, cpus)
    requested = max(1, int(config.requested_workers))
    cap = max(1, min(int(config.max_workers), cpu_cap, requested))
    mem = _available_memory_mb()
    if mem is not None and config.min_free_mem_mb_per_worker > 0:
        mem_cap = max(1, int(mem // config.min_free_mem_mb_per_worker))
        cap = min(cap, mem_cap)
    return max(1, cap)


class LocalDistributedEvaluator:
    """Process-pool evaluator with conservative resource caps."""

    def __init__(self, config: Optional[DistributedEvalConfig] = None):
        self.config = config or DistributedEvalConfig()

    def map(self, fn: Callable[[Any], Any], inputs: Sequence[Any]) -> DistributedEvalResult:
        started = time.time()
        items = list(inputs)
        if not items:
            return DistributedEvalResult(worker_count=0, input_count=0, outputs=[], runtime_s=0.0)
        workers = min(safe_worker_count(self.config), len(items))
        errors: List[str] = []
        outputs: List[Any] = []
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
                for value in pool.map(fn, items):
                    outputs.append(value)
        except Exception as exc:
            errors.append(f"{type(exc).__name__}:{exc}")
        return DistributedEvalResult(
            worker_count=workers,
            input_count=len(items),
            outputs=outputs,
            runtime_s=round(time.time() - started, 6),
            errors=errors,
        )


__all__ = [
    "DistributedEvalConfig",
    "DistributedEvalResult",
    "LocalDistributedEvaluator",
    "safe_worker_count",
]
