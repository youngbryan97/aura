from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from core.runtime.resource_observation import ResourceObserver, get_resource_observer

from .types import clamp01

logger = logging.getLogger("Aura.Morphogenesis.Metabolism")


@dataclass
class ResourceSnapshot:
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_available_mb: float = 0.0
    load_average_1m: float = 0.0
    pressure: float = 0.0
    observation_source: str = "unavailable"
    observation_scenario_id: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cpu_percent": round(float(self.cpu_percent), 3),
            "memory_percent": round(float(self.memory_percent), 3),
            "memory_used_mb": round(float(self.memory_used_mb), 1),
            "memory_available_mb": round(float(self.memory_available_mb), 1),
            "load_average_1m": round(float(self.load_average_1m), 3),
            "pressure": round(float(self.pressure), 5),
            "observation_source": self.observation_source,
            "observation_scenario_id": self.observation_scenario_id,
            "timestamp": self.timestamp,
        }


@dataclass
class CellBudget:
    cell_id: str
    priority: float = 0.5
    energy: float = 0.5
    max_energy: float = 1.0
    spent_total: float = 0.0
    denied_count: int = 0
    last_used: float = field(default_factory=time.time)

    def can_spend(self, amount: float) -> bool:
        return self.energy >= max(0.0, amount)

    def spend(self, amount: float) -> bool:
        amount = max(0.0, float(amount))
        if self.energy < amount:
            self.denied_count += 1
            return False
        self.energy = clamp01(self.energy - amount)
        self.spent_total += amount
        self.last_used = time.time()
        return True

    def recover(self, amount: float) -> None:
        self.energy = min(float(self.max_energy), max(0.0, self.energy + amount))


class MetabolismManager:
    """Resource budget manager for morphogenetic cells.

    This prevents the biological metaphor from turning into cancer:
    no runaway replication, no unlimited work, no unowned task storm.
    """

    def __init__(
        self,
        *,
        global_energy: float = 1.0,
        recovery_per_tick: float = 0.035,
        high_pressure_threshold: float = 0.82,
        observer: ResourceObserver | None = None,
    ):
        self.global_energy = clamp01(global_energy)
        self.recovery_per_tick = clamp01(recovery_per_tick)
        self.high_pressure_threshold = clamp01(high_pressure_threshold)
        self._observer = observer
        self._budgets: dict[str, CellBudget] = {}
        self._last_snapshot = ResourceSnapshot()

    def ensure_budget(self, cell_id: str, *, priority: float = 0.5, baseline: float = 0.35, max_energy: float = 1.0) -> CellBudget:
        b = self._budgets.get(cell_id)
        if b is None:
            b = CellBudget(
                cell_id=cell_id,
                priority=clamp01(priority),
                energy=clamp01(baseline),
                max_energy=max(0.01, float(max_energy)),
            )
            self._budgets[cell_id] = b
        return b

    def spend(self, cell_id: str, amount: float) -> bool:
        b = self.ensure_budget(cell_id)
        if self.global_energy < amount * 0.35:
            b.denied_count += 1
            return False
        ok = b.spend(amount)
        if ok:
            self.global_energy = clamp01(self.global_energy - amount * 0.08)
        return ok

    def pulse(self) -> ResourceSnapshot:
        snap = self.sample_resources()
        self._last_snapshot = snap
        pressure = snap.pressure

        # Recover more slowly under pressure. Protected/high priority cells
        # get proportionally more energy.
        global_recovery = self.recovery_per_tick * (1.0 - pressure)
        self.global_energy = clamp01(self.global_energy + global_recovery)

        for b in self._budgets.values():
            recovery = self.recovery_per_tick * (0.35 + b.priority) * (1.0 - pressure)
            b.recover(recovery)
        return snap

    def sample_resources(self) -> ResourceSnapshot:
        observer = self._observer or get_resource_observer()
        provenance = observer.provenance
        try:
            memory = observer.memory()
            compute = observer.compute()
            cpu = float(compute.cpu_percent)
            used = float(memory.used_bytes) / float(1024**2)
            avail = float(memory.available_bytes) / float(1024**2)
            mem_pct = float(memory.percent) if memory.available else 100.0
            load_1 = float(compute.load_1m)
            cpu_count = max(1, int(compute.cpu_count))
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            cpu, used, avail, mem_pct, load_1, cpu_count = 100.0, 0.0, 0.0, 100.0, 0.0, 1

        pressure = clamp01(
            max(
                cpu / 100.0,
                mem_pct / 100.0,
                min(1.0, load_1 / cpu_count),
            )
        )
        return ResourceSnapshot(
            cpu_percent=cpu,
            memory_percent=mem_pct,
            memory_used_mb=used,
            memory_available_mb=avail,
            load_average_1m=load_1,
            pressure=pressure,
            observation_source=provenance.source.value,
            observation_scenario_id=provenance.scenario_id,
        )

    @property
    def high_pressure(self) -> bool:
        return self._last_snapshot.pressure >= self.high_pressure_threshold

    def status(self) -> dict[str, Any]:
        return {
            "global_energy": round(float(self.global_energy), 5),
            "high_pressure": self.high_pressure,
            "resources": self._last_snapshot.to_dict(),
            "budgets": {
                cid: {
                    "energy": round(float(b.energy), 5),
                    "priority": b.priority,
                    "spent_total": round(float(b.spent_total), 5),
                    "denied_count": b.denied_count,
                }
                for cid, b in self._budgets.items()
            },
        }
