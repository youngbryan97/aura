"""Hardware feasibility checks for Aura's local-model claims.

The point of this module is deliberately unromantic: it gives the runtime and
the tests a way to say "this configuration is a simulation/prototype on this
machine" instead of laundering a memory-bound setup into a real-time autonomy
claim.  It is intentionally conservative because optimistic memory math is how
32B-on-16GB claims become theater.
"""
from __future__ import annotations


from dataclasses import dataclass, field
from typing import Iterable


_GIB = 1024**3


@dataclass(frozen=True)
class ModelMemoryProfile:
    """Approximate local inference memory for one model tier."""

    name: str
    parameters_b: float
    quantization_bits: int
    context_tokens: int = 4096
    hidden_size: int = 5120
    layers: int = 40
    activation_overhead_gib: float = 1.25
    runtime_overhead_gib: float = 1.5

    @property
    def weight_gib(self) -> float:
        return (self.parameters_b * 1_000_000_000 * self.quantization_bits / 8) / _GIB

    @property
    def kv_cache_gib(self) -> float:
        # K and V tensors, fp16/bf16 cache.  Architectures differ, but this is
        # close enough to expose memory pressure without pretending precision.
        bytes_ = self.context_tokens * self.layers * self.hidden_size * 2 * 2
        return bytes_ / _GIB

    @property
    def total_gib(self) -> float:
        return (
            self.weight_gib
            + self.kv_cache_gib
            + self.activation_overhead_gib
            + self.runtime_overhead_gib
        )


@dataclass(frozen=True)
class WorkloadProfile:
    """Memory and latency-sensitive non-LLM work that shares the machine."""

    mesh_neurons: int = 4096
    modules: int = 90
    substrate_overhead_gib: float = 0.65
    database_cache_gib: float = 0.25
    os_reserved_gib: float = 4.0
    safety_margin_gib: float = 2.0

    @property
    def total_gib(self) -> float:
        return (
            self.substrate_overhead_gib
            + self.database_cache_gib
            + self.os_reserved_gib
            + self.safety_margin_gib
        )


@dataclass(frozen=True)
class MachineProfile:
    name: str
    unified_memory_gib: float
    gpu: str = "unknown"
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class HardwareVerdict:
    machine: MachineProfile
    model: ModelMemoryProfile
    workload: WorkloadProfile
    required_gib: float
    headroom_gib: float
    feasible: bool
    realtime_heartbeat_feasible: bool
    classification: str
    warnings: tuple[str, ...] = field(default_factory=tuple)
    recommended_tier: str = "unknown"

    def as_dict(self) -> dict[str, object]:
        return {
            "machine": self.machine.name,
            "unified_memory_gib": round(self.machine.unified_memory_gib, 2),
            "model": self.model.name,
            "model_required_gib": round(self.model.total_gib, 2),
            "workload_required_gib": round(self.workload.total_gib, 2),
            "required_gib": round(self.required_gib, 2),
            "headroom_gib": round(self.headroom_gib, 2),
            "feasible": self.feasible,
            "realtime_heartbeat_feasible": self.realtime_heartbeat_feasible,
            "classification": self.classification,
            "warnings": list(self.warnings),
            "recommended_tier": self.recommended_tier,
        }


def default_model_tiers() -> list[ModelMemoryProfile]:
    return [
        ModelMemoryProfile(
            name="32B-8bit",
            parameters_b=32.0,
            quantization_bits=8,
            hidden_size=5120,
            layers=64,
            activation_overhead_gib=2.0,
            runtime_overhead_gib=2.0,
        ),
        ModelMemoryProfile(
            name="32B-4bit",
            parameters_b=32.0,
            quantization_bits=4,
            hidden_size=5120,
            layers=64,
            activation_overhead_gib=1.6,
            runtime_overhead_gib=1.8,
        ),
        ModelMemoryProfile(
            name="7B-4bit",
            parameters_b=7.0,
            quantization_bits=4,
            hidden_size=4096,
            layers=32,
            activation_overhead_gib=0.8,
            runtime_overhead_gib=1.0,
        ),
        ModelMemoryProfile(
            name="1.5B-4bit",
            parameters_b=1.5,
            quantization_bits=4,
            hidden_size=2048,
            layers=24,
            activation_overhead_gib=0.45,
            runtime_overhead_gib=0.7,
        ),
    ]


class HardwareRealityAuditor:
    """Conservative checker for memory-bound model claims."""

    def __init__(
        self,
        machine: MachineProfile,
        workload: WorkloadProfile | None = None,
        tiers: Iterable[ModelMemoryProfile] | None = None,
    ) -> None:
        self.machine = machine
        self.workload = workload or WorkloadProfile()
        self.tiers = list(tiers or default_model_tiers())

    def evaluate(self, model: ModelMemoryProfile) -> HardwareVerdict:
        required = model.total_gib + self.workload.total_gib
        headroom = self.machine.unified_memory_gib - required
        feasible = headroom >= 0
        realtime = feasible and headroom >= 3.0 and model.parameters_b <= 8.0

        warnings: list[str] = []
        if not feasible:
            warnings.append(
                "required memory exceeds available unified memory; expect swap, cloud offload, or failure"
            )
        elif headroom < 3.0:
            warnings.append(
                "configuration barely fits; do not call it real-time without measured latency logs"
            )
        if model.parameters_b >= 30.0 and self.machine.unified_memory_gib <= 32.0:
            warnings.append(
                "32B local inference on <=32 GiB unified memory is a high-pressure batch tier, not a heartbeat tier"
            )
        if not realtime:
            warnings.append(
                "continuous 100-500 ms heartbeat claims require a smaller model tier or measured external evidence"
            )

        recommended = self.recommend_tier().name
        if not feasible:
            classification = "not_feasible"
        elif realtime:
            classification = "heartbeat_candidate"
        else:
            classification = "batch_or_high_level_cortex"

        return HardwareVerdict(
            machine=self.machine,
            model=model,
            workload=self.workload,
            required_gib=required,
            headroom_gib=headroom,
            feasible=feasible,
            realtime_heartbeat_feasible=realtime,
            classification=classification,
            warnings=tuple(warnings),
            recommended_tier=recommended,
        )

    def evaluate_all(self) -> list[HardwareVerdict]:
        return [self.evaluate(tier) for tier in self.tiers]

    def recommend_tier(self) -> ModelMemoryProfile:
        feasible = [verdict for verdict in self.evaluate_all_shallow() if verdict[1] >= 3.0]
        if feasible:
            return feasible[0][0]
        return self.tiers[-1]

    def evaluate_all_shallow(self) -> list[tuple[ModelMemoryProfile, float]]:
        results: list[tuple[ModelMemoryProfile, float]] = []
        for tier in self.tiers:
            required = tier.total_gib + self.workload.total_gib
            results.append((tier, self.machine.unified_memory_gib - required))
        return results


def m1_pro_16gb_profile() -> MachineProfile:
    return MachineProfile(
        name="Apple M1 Pro 16GB",
        unified_memory_gib=16.0,
        gpu="Apple Metal unified-memory GPU",
        notes=("shared CPU/GPU memory", "swap pressure affects inference latency"),
    )

