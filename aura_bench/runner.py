"""aura_bench.runner

Pre-registered benchmark runner. Each registered test produces a
``BenchResult`` with the pre-registered hypothesis, metric, threshold, the
observed value, the number of trials, the baseline run, the ablation run,
and a verdict (pass/fail/inconclusive).

The runner refuses to record a verdict that wasn't *pre-registered* —
``test.declare()`` must be called with hypothesis + metric + threshold
*before* ``test.run()``. This is the structural guard against post-hoc
"this counts because it passed".

Usage:

    from aura_bench.runner import register, run_all

    @register
    class MyTest(BenchTest):
        name = "phi_uncurated_positive"
        async def declare(self): ...
        async def run(self) -> Sample: ...
        async def baseline(self) -> Sample: ...
        async def ablation(self) -> Sample: ...

    asyncio.run(run_all())
"""
from core.runtime.atomic_writer import atomic_write_text
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Type

logger = logging.getLogger("Aura.Bench")

_BENCH_DIR = Path.home() / ".aura" / "data" / "bench"
_BENCH_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Registration:
    hypothesis: str
    metric: str
    pass_threshold: float  # observed metric must >= threshold
    trials: int
    baseline_label: str
    ablation_label: str
    declared_at: float = field(default_factory=time.time)


@dataclass
class Sample:
    metric: float
    detail: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchResult:
    name: str
    registration: Registration
    full: Sample
    baseline: Optional[Sample] = None
    ablation: Optional[Sample] = None
    verdict: str = "inconclusive"
    notes: str = ""
    finished_at: float = field(default_factory=time.time)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "registration": asdict(self.registration),
            "full": asdict(self.full),
            "baseline": asdict(self.baseline) if self.baseline else None,
            "ablation": asdict(self.ablation) if self.ablation else None,
            "verdict": self.verdict,
            "notes": self.notes,
            "finished_at": self.finished_at,
        }


class BenchTest:
    name: str = "unnamed"

    async def declare(self) -> Registration:  # pragma: no cover - subclass
        raise NotImplementedError

    async def run(self) -> Sample:  # pragma: no cover - subclass
        raise NotImplementedError

    async def baseline(self) -> Optional[Sample]:
        return None

    async def ablation(self) -> Optional[Sample]:
        return None


_REGISTRY: List[Type[BenchTest]] = []


def register(cls: Type[BenchTest]) -> Type[BenchTest]:
    _REGISTRY.append(cls)
    return cls


def _verdict(reg: Registration, full: Sample, baseline: Optional[Sample], ablation: Optional[Sample]) -> str:
    if full.metric < reg.pass_threshold:
        return "fail_threshold"
    if baseline is not None and full.metric < baseline.metric:
        return "fail_baseline"
    if ablation is not None and full.metric <= ablation.metric:
        return "fail_ablation"
    return "pass"


async def run_one(test: BenchTest) -> BenchResult:
    reg = await test.declare()
    full = await test.run()
    baseline = await test.baseline()
    ablation = await test.ablation()
    res = BenchResult(
        name=test.name,
        registration=reg,
        full=full,
        baseline=baseline,
        ablation=ablation,
        verdict=_verdict(reg, full, baseline, ablation),
    )
    _persist(res)
    return res


async def run_all() -> List[BenchResult]:
    out: List[BenchResult] = []
    for cls in _REGISTRY:
        try:
            res = await run_one(cls())
        except Exception as exc:
            logger.exception("bench test %s crashed", cls.__name__)
            reg = Registration(hypothesis="crash", metric="-", pass_threshold=0.0, trials=0, baseline_label="-", ablation_label="-")
            res = BenchResult(name=cls.name, registration=reg, full=Sample(metric=0.0), verdict="error", notes=str(exc))
            _persist(res)
        out.append(res)
    return out


def _persist(res: BenchResult) -> None:
    out = _BENCH_DIR / "results.jsonl"
    try:
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(res.as_dict(), default=str) + "\n")
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except Exception:
                pass
    except Exception as exc:
        logger.warning("bench persist failed: %s", exc)


def write_report(results: List[BenchResult]) -> Path:
    out = _BENCH_DIR / "report.md"
    lines = ["# aura_bench report", "", f"_generated {time.strftime('%Y-%m-%d %H:%M:%S')}_", ""]
    for r in results:
        lines.append(f"## {r.name} — **{r.verdict}**")
        lines.append("")
        lines.append(f"- hypothesis: {r.registration.hypothesis}")
        lines.append(f"- metric: `{r.registration.metric}`")
        lines.append(f"- threshold: `{r.registration.pass_threshold}`")
        lines.append(f"- trials: {r.registration.trials}")
        lines.append(f"- full: `{r.full.metric}`")
        if r.baseline:
            lines.append(f"- baseline: `{r.baseline.metric}`")
        if r.ablation:
            lines.append(f"- ablation: `{r.ablation.metric}`")
        if r.notes:
            lines.append(f"- notes: {r.notes}")
        lines.append("")
    atomic_write_text(out, "\n".join(lines), encoding="utf-8")
    return out


__all__ = [
    "Registration",
    "Sample",
    "BenchResult",
    "BenchTest",
    "register",
    "run_one",
    "run_all",
    "write_report",
]
