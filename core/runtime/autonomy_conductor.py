"""Active scheduler for Aura's self-maintenance loops.

This is the difference between machinery existing and Aura actually using it.
The conductor owns recurring jobs, records receipts, and marks missed or failed
runs as degraded events.  It is lightweight enough for desktop runtime and
safe enough to start automatically.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import record_degradation

JobFn = Callable[[], Awaitable[dict[str, Any]] | dict[str, Any]]


@dataclass
class ConductedJob:
    name: str
    interval_s: float
    fn: JobFn
    run_immediately: bool = False
    last_started_at: float = 0.0
    last_finished_at: float = 0.0
    last_status: str = "never_run"
    last_result: dict[str, Any] = field(default_factory=dict)
    failures: int = 0

    def due(self, now: float) -> bool:
        if self.last_started_at <= 0:
            return self.run_immediately
        return (now - self.last_started_at) >= self.interval_s

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "interval_s": self.interval_s,
            "last_started_at": self.last_started_at,
            "last_finished_at": self.last_finished_at,
            "last_status": self.last_status,
            "last_result": self.last_result,
            "failures": self.failures,
        }


class AutonomyConductor:
    """Runs self-maintenance jobs consistently with observable receipts."""

    def __init__(self, ledger_path: str | Path | None = None) -> None:
        self.ledger_path = Path(ledger_path or Path.home() / ".aura" / "data" / "runtime" / "autonomy_conductor.jsonl")
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, ConductedJob] = {}
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    def register(self, name: str, interval_s: float, fn: JobFn, *, run_immediately: bool = False) -> None:
        self.jobs[name] = ConductedJob(name=name, interval_s=float(interval_s), fn=fn, run_immediately=run_immediately)

    def register_defaults(self) -> None:
        self.register("metabolic_budget", 300.0, self._job_metabolic_budget, run_immediately=True)
        self.register("stdp_external_validation", 6 * 3600.0, self._job_stdp_external_validation, run_immediately=True)
        self.register("caa_32b_validation", 6 * 3600.0, self._job_caa_32b_validation, run_immediately=True)
        self.register("proof_bundle", 12 * 3600.0, self._job_proof_bundle, run_immediately=False)
        self.register("self_test_synthesis", 24 * 3600.0, self._job_self_test_synthesis, run_immediately=False)

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        if not self.jobs:
            self.register_defaults()
        self._stop = asyncio.Event()
        self._task = asyncio.create_task(self._run(), name="Aura.AutonomyConductor")

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            await asyncio.wait([self._task], timeout=3)

    async def run_due_once(self) -> dict[str, Any]:
        now = time.time()
        results: dict[str, Any] = {}
        for job in self.jobs.values():
            if job.due(now):
                results[job.name] = await self._run_job(job)
        return results

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                await self.run_due_once()
            except Exception as exc:
                record_degradation("autonomy_conductor", exc)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=30.0)
            except asyncio.TimeoutError:
                pass

    async def _run_job(self, job: ConductedJob) -> dict[str, Any]:
        job.last_started_at = time.time()
        try:
            result = job.fn()
            if asyncio.iscoroutine(result):
                result = await result
            job.last_result = dict(result or {})
            job.last_status = "ok"
            job.failures = 0
        except Exception as exc:
            record_degradation("autonomy_conductor", exc)
            job.last_result = {"error": repr(exc)}
            job.last_status = "failed"
            job.failures += 1
        job.last_finished_at = time.time()
        self._record(job)
        return job.to_dict()

    def _record(self, job: ConductedJob) -> None:
        entry = {"when": time.time(), "job": job.to_dict()}
        with open(self.ledger_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, sort_keys=True, default=str) + "\n")

    def status(self) -> dict[str, Any]:
        return {
            "active": bool(self._task and not self._task.done()),
            "jobs": {name: job.to_dict() for name, job in sorted(self.jobs.items())},
            "ledger_path": str(self.ledger_path),
        }

    def write_status(self, path: str | Path) -> dict[str, Any]:
        status = self.status()
        atomic_write_text(Path(path), json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
        return status

    async def _job_metabolic_budget(self) -> dict[str, Any]:
        from core.autonomy.metabolic_budget import MetabolicState, get_metabolic_budget_scheduler

        allocation = get_metabolic_budget_scheduler().allocate(
            MetabolicState(stability=0.9, resource_headroom=0.8, novelty_budget=0.6, benchmark_gap=0.25, external_usefulness=0.6)
        )
        return allocation.to_dict()

    async def _job_stdp_external_validation(self) -> dict[str, Any]:
        from core.consciousness.stdp_external_validation import STDPExternalValidator

        return STDPExternalValidator().run(steps=64).to_dict()

    async def _job_caa_32b_validation(self) -> dict[str, Any]:
        from training.caa_32b_validation import CAA32BValidator

        return CAA32BValidator().run()

    async def _job_proof_bundle(self) -> dict[str, Any]:
        from tools.proof_bundle import build_proof_bundle

        output = Path.home() / ".aura" / "data" / "proof_bundle" / "latest"
        return build_proof_bundle(output_dir=output)

    async def _job_self_test_synthesis(self) -> dict[str, Any]:
        from core.evaluation.self_test_synthesizer import SelfTestSynthesizer

        synth = SelfTestSynthesizer()
        tests = synth.synthesize_tests([])
        return {"generated_tests": len(tests)}


_instance: AutonomyConductor | None = None


def get_autonomy_conductor() -> AutonomyConductor:
    global _instance
    if _instance is None:
        _instance = AutonomyConductor()
    return _instance


async def start_default_conductor() -> AutonomyConductor:
    conductor = get_autonomy_conductor()
    conductor.register_defaults()
    await conductor.start()
    return conductor


__all__ = ["ConductedJob", "AutonomyConductor", "get_autonomy_conductor", "start_default_conductor"]
