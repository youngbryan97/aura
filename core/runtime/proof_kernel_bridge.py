"""Live bridge from Aura runtime state into the standalone proof kernel.

The proof kernel stays honest about its claim scope, but it is no longer a
synthetic sidecar only. This bridge samples live runtime evidence, runs the
same homeostasis/global-workspace primitives when available, and registers a
receipt-like snapshot that activation audit can verify.
"""
from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation


@dataclass
class ProofKernelRuntimeSnapshot:
    generated_at: float
    live_inputs: dict[str, Any]
    proof_metrics: dict[str, Any]
    claim_scope: dict[str, list[str]]
    active: bool
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ProofKernelBridge:
    """Runs bounded proof-kernel probes over live Aura runtime evidence."""

    def __init__(self) -> None:
        self.last_snapshot: ProofKernelRuntimeSnapshot | None = None

    async def sample(self) -> ProofKernelRuntimeSnapshot:
        live_inputs = self._collect_live_inputs()
        errors: list[str] = []
        proof_metrics: dict[str, Any] = {}
        try:
            kernel = self._load_kernel()
            signals = kernel.ExternalSignals(
                health_error_rate=float(live_inputs.get("degradation_pressure", 0.0)),
                resource_anxiety=float(live_inputs.get("resource_pressure", 0.0)),
                thermal_load=float(live_inputs.get("thermal_pressure", 0.0)),
                sovereignty_score=float(live_inputs.get("governance_score", 1.0)),
            )
            homeostasis = kernel.HomeostasisEngine()
            proof_metrics["homeostasis"] = await homeostasis.pulse(signals)
            workspace = kernel.GlobalWorkspace()
            await workspace.submit(
                kernel.CognitiveCandidate(
                    content="runtime evidence sample",
                    source="proof_kernel_bridge",
                    priority=max(0.2, 1.0 - float(live_inputs.get("degradation_pressure", 0.0))),
                    content_type=kernel.ContentType.META,
                )
            )
            winner = await workspace.run_competition()
            proof_metrics["workspace"] = workspace.get_snapshot()
            proof_metrics["winner_source"] = winner.source if winner else None
        except Exception as exc:
            record_degradation("proof_kernel_bridge", exc, severity="warning", action="reported bridge snapshot as degraded")
            errors.append(f"{type(exc).__name__}: {exc}")

        snapshot = ProofKernelRuntimeSnapshot(
            generated_at=time.time(),
            live_inputs=live_inputs,
            proof_metrics=proof_metrics,
            claim_scope={
                "supports": [
                    "live runtime evidence is sampled",
                    "homeostatic and attention proxies are computed over live inputs",
                    "activation audit can prove the bridge is registered",
                ],
                "does_not_support": [
                    "proof of subjective experience",
                    "LLM-final proof obligations",
                    "environment mastery by itself",
                ],
            },
            active=not errors and bool(proof_metrics),
            errors=errors,
        )
        self.last_snapshot = snapshot
        return snapshot

    def status(self) -> dict[str, Any]:
        if self.last_snapshot is None:
            return {"active": False, "reason": "not_sampled"}
        return self.last_snapshot.to_dict()

    @staticmethod
    def _load_kernel() -> Any:
        try:
            import aura_consciousness_proof as kernel  # type: ignore

            return kernel
        except Exception:
            root = Path(__file__).resolve().parents[2] / "proof_kernel" / "src"
            if root.exists() and str(root) not in sys.path:
                sys.path.insert(0, str(root))
            import aura_consciousness_proof as kernel  # type: ignore

            return kernel

    @staticmethod
    def _collect_live_inputs() -> dict[str, Any]:
        inputs: dict[str, Any] = {
            "degradation_pressure": 0.0,
            "resource_pressure": 0.0,
            "thermal_pressure": 0.0,
            "governance_score": 1.0,
        }
        try:
            from core.runtime.errors import get_degradation_tracker

            recent = get_degradation_tracker().recent(limit=50)
            critical = sum(1 for rec in recent if rec.severity == "critical")
            degraded = sum(1 for rec in recent if rec.severity in {"degraded", "warning"})
            inputs["degradation_pressure"] = min(1.0, (critical * 0.2 + degraded * 0.05))
        except Exception as exc:
            record_degradation("proof_kernel_bridge", exc, severity="debug", action="omitted degradation pressure")
        try:
            from core.container import ServiceContainer

            perf = ServiceContainer.get("performance_guard", default=None)
            status = perf.status() if perf and hasattr(perf, "status") else {}
            if isinstance(status, dict):
                pressure = status.get("pressure") or status.get("resource_pressure") or {}
                if isinstance(pressure, dict):
                    inputs["resource_pressure"] = float(pressure.get("memory", 0.0) or pressure.get("ram", 0.0) or 0.0)
                    inputs["thermal_pressure"] = float(pressure.get("thermal", 0.0) or 0.0)
            will = ServiceContainer.get("unified_will", default=None) or ServiceContainer.get("will", default=None)
            inputs["governance_score"] = 1.0 if will is not None else 0.85
        except Exception as exc:
            record_degradation("proof_kernel_bridge", exc, severity="debug", action="used default live inputs")
        return inputs


_bridge: ProofKernelBridge | None = None


def get_proof_kernel_bridge() -> ProofKernelBridge:
    global _bridge
    if _bridge is None:
        _bridge = ProofKernelBridge()
    return _bridge


async def start_proof_kernel_bridge() -> ProofKernelBridge:
    bridge = get_proof_kernel_bridge()
    await asyncio.wait_for(bridge.sample(), timeout=5.0)
    try:
        from core.container import ServiceContainer

        ServiceContainer.register_instance("proof_kernel_bridge", bridge, required=False)
    except Exception as exc:
        record_degradation("proof_kernel_bridge", exc, severity="debug", action="sampled but did not register")
    return bridge


__all__ = [
    "ProofKernelBridge",
    "ProofKernelRuntimeSnapshot",
    "get_proof_kernel_bridge",
    "start_proof_kernel_bridge",
]
