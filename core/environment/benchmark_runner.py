"""Structured benchmark runner for evaluating policies."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Any

from core.environment.environment_kernel import EnvironmentKernel


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    run_id: str
    baseline_name: str
    success: bool
    total_steps: int
    final_score: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)


class BenchmarkRunner:
    """Executes the environment kernel against different policy baselines."""

    def __init__(self, kernel_factory: Callable[[], EnvironmentKernel]):
        self.kernel_factory = kernel_factory
        self.log = logging.getLogger(__name__)

    async def run_baseline(
        self, 
        baseline_name: str, 
        policy_fn: Callable[[EnvironmentKernel], Any], 
        max_steps: int = 1000
    ) -> BenchmarkResult:
        """Executes a full run using the provided policy function."""
        run_id = f"bench_{baseline_name}_{int(time.time())}"
        kernel = self.kernel_factory()
        
        self.log.info(f"Starting benchmark run {run_id} using baseline {baseline_name}")
        await kernel.start(run_id=run_id)
        
        steps = 0
        success = False
        
        try:
            while steps < max_steps:
                # The policy function determines the intent and steps the kernel
                # e.g., intent = policy_fn(kernel); frame = await kernel.step(intent)
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
            
        result = BenchmarkResult(
            run_id=run_id,
            baseline_name=baseline_name,
            success=success,
            total_steps=steps,
        )
        self.log.info(f"Benchmark run {run_id} completed: {result}")
        return result

__all__ = ["BenchmarkResult", "BenchmarkRunner"]
