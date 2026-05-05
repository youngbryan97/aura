"""Structured benchmark runner for evaluating policies.

Enforces strict mode separation:
- strict_real: actual environment, full boundary guard, real scores
- simulated_canary: deterministic fake adapter, canary-only scores
- fixture_replay: recorded screens, parser/policy regression only
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Any, Literal

from core.environment.environment_kernel import EnvironmentKernel
from core.environment.boundary_guard import BoundaryGuard, IntegrityReport

RunMode = Literal["strict_real", "simulated_canary", "fixture_replay"]


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    run_id: str
    baseline_name: str
    mode: RunMode
    success: bool
    total_steps: int
    final_score: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)
    integrity_report: IntegrityReport | None = None
    simulated: bool = False
    contaminated: bool = False


@dataclass 
class BenchmarkReport:
    """Aggregate report across baselines and ablations."""
    results: list[BenchmarkResult] = field(default_factory=list)
    baselines: dict[str, BenchmarkResult] = field(default_factory=dict)
    ablations: dict[str, BenchmarkResult] = field(default_factory=dict)

    def real_results(self) -> list[BenchmarkResult]:
        """Only results from strict_real mode, not contaminated."""
        return [r for r in self.results if r.mode == "strict_real" and not r.contaminated]

    def canary_results(self) -> list[BenchmarkResult]:
        return [r for r in self.results if r.mode == "simulated_canary"]

    def add_result(self, result: BenchmarkResult) -> None:
        self.results.append(result)
        if result.baseline_name.startswith("ablation_"):
            self.ablations[result.baseline_name] = result
        else:
            self.baselines[result.baseline_name] = result


class BenchmarkRunner:
    """Executes the environment kernel against different policy baselines."""

    def __init__(
        self,
        kernel_factory: Callable[[], EnvironmentKernel],
        mode: RunMode = "strict_real",
    ):
        self.kernel_factory = kernel_factory
        self.mode = mode
        self.log = logging.getLogger(__name__)
        self.boundary_guard = BoundaryGuard()
        self.report = BenchmarkReport()

    async def run_baseline(
        self,
        baseline_name: str,
        policy_fn: Callable[[EnvironmentKernel], Any],
        max_steps: int = 1000,
    ) -> BenchmarkResult:
        """Executes a full run using the provided policy function."""
        run_id = f"bench_{baseline_name}_{int(time.time())}"
        kernel = self.kernel_factory()

        self.log.info(f"Starting benchmark run {run_id} mode={self.mode} baseline={baseline_name}")
        await kernel.start(run_id=run_id)

        steps = 0
        success = False

        try:
            while steps < max_steps:
                intent = await policy_fn(kernel)
                frame = await kernel.step(intent)
                steps += 1

                # Check terminal state
                if kernel.episode and kernel.episode.terminal:
                    success = kernel.episode.success if hasattr(kernel.episode, 'success') else False
                    break
        except Exception as e:
            self.log.error(f"Benchmark run {run_id} failed with error: {e}")
        finally:
            await kernel.close()

        # Generate integrity report
        integrity = self.boundary_guard.get_integrity_report(run_id, self.mode)

        result = BenchmarkResult(
            run_id=run_id,
            baseline_name=baseline_name,
            mode=self.mode,
            success=success,
            total_steps=steps,
            integrity_report=integrity,
            simulated=self.mode == "simulated_canary",
            contaminated=self.boundary_guard.contaminated,
        )
        self.report.add_result(result)
        self.log.info(f"Benchmark run {run_id} completed: steps={steps} success={success}")
        return result

    def validate_for_deep_run_claim(self) -> tuple[bool, list[str]]:
        """Check if results can support a deep-run claim."""
        failures = []
        real = self.report.real_results()
        if not real:
            failures.append("No strict_real results")

        for r in real:
            if r.integrity_report and r.integrity_report.verdict != "CLEAN":
                failures.append(f"Run {r.run_id} has contaminated integrity report")
            if r.simulated:
                failures.append(f"Run {r.run_id} marked as simulated but in real results")

        return len(failures) == 0, failures


__all__ = ["BenchmarkResult", "BenchmarkReport", "BenchmarkRunner", "RunMode"]

