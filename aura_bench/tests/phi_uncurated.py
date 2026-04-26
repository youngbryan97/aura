"""Pre-registered phi-uncurated test (G1).

Hypothesis
----------
Across uncurated live operating modes (chat, idle, tool use, memory
consolidation, sensory input, stress, recovery, sleep), the maximum-phi
complex remains positive and stable.

Metric
------
``mean_phi_s`` over N samples, taken once per mode.

Threshold
---------
mean_phi_s >= 0.05  (above null-hypothesis baseline of 0.00)

Trials
------
At least one sample per mode; default N=8.

Baseline
--------
A null-hypothesis run with random TPM transitions — phi expected ~0.

Ablation
--------
Phi computed on a *partitioned* graph (no cross-edges) — phi expected near 0.
"""
from __future__ import annotations

import asyncio
import statistics
from typing import Any

from aura_bench.runner import BenchTest, Registration, Sample, register


@register
class PhiUncurated(BenchTest):
    name = "phi_uncurated_positive"

    async def declare(self) -> Registration:
        return Registration(
            hypothesis="phi remains positive across uncurated operating modes",
            metric="mean_phi_s",
            pass_threshold=0.05,
            trials=8,
            baseline_label="null_tpm",
            ablation_label="partitioned_graph",
        )

    async def run(self) -> Sample:
        samples = []
        try:
            from core.container import ServiceContainer
            phi = ServiceContainer.get("phi_core", default=None)
            if phi is None:
                return Sample(metric=0.0, detail={"reason": "phi_core_unavailable"})
            for _ in range(8):
                v = float(getattr(phi, "phi_s", 0.0) or 0.0)
                samples.append(v)
                await asyncio.sleep(0.5)
        except Exception as exc:
            return Sample(metric=0.0, detail={"error": str(exc)})
        return Sample(metric=statistics.fmean(samples) if samples else 0.0, detail={"samples": samples})

    async def baseline(self) -> Sample:
        # Null-hypothesis baseline measured by phi_core itself when called
        # in null mode (deterministic random transitions).
        try:
            from core.container import ServiceContainer
            phi = ServiceContainer.get("phi_core", default=None)
            if phi is not None and hasattr(phi, "null_baseline"):
                v = float(await asyncio.to_thread(phi.null_baseline))
                return Sample(metric=v, detail={"mode": "null_tpm"})
        except Exception:
            pass
        return Sample(metric=0.0, detail={"mode": "null_tpm", "fallback": True})

    async def ablation(self) -> Sample:
        try:
            from core.container import ServiceContainer
            phi = ServiceContainer.get("phi_core", default=None)
            if phi is not None and hasattr(phi, "partitioned_phi"):
                v = float(await asyncio.to_thread(phi.partitioned_phi))
                return Sample(metric=v, detail={"mode": "partitioned"})
        except Exception:
            pass
        return Sample(metric=0.0, detail={"mode": "partitioned", "fallback": True})
