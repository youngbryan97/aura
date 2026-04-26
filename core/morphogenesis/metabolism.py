from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .types import clamp01, json_safe

logger = logging.getLogger("Aura.Morphogenesis.Metabolism")


@dataclass
class ResourceSnapshot:
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_available_mb: float = 0.0
    load_average_1m: float = 0.0
    pressure: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cpu_percent": round(float(self.cpu_percent), 3),
            "memory_percent": round(float(self.memory_percent), 3),
            "memory_used_mb": round(float(self.memory_used_mb), 1),
            "memory_available_mb": round(float(self.memory_available_mb), 1),
            "load_average_1m": round(float(self.load_average_1m), 3),
            "pressure": round(float(self.pressure), 5),
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
    ):
        self.global_energy = clamp01(global_energy)
        self.recovery_per_tick = clamp01(recovery_per_tick)
        self.high_pressure_threshold = clamp01(high_pressure_threshold)
        self._budgets: Dict[str, CellBudget] = {}
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
        try:
            import psutil
            mem = psutil.virtual_memory()
            cpu = float(psutil.cpu_percent(interval=None))
            used = float(getattr(mem, "used", 0.0)) / (1024 ** 2)
            avail = float(getattr(mem, "available", 0.0)) / (1024 ** 2)
            mem_pct = float(getattr(mem, "percent", 0.0))
        except Exception:
            cpu, used, avail, mem_pct = 0.0, 0.0, 0.0, 0.0

        try:
            load_1 = float(os.getloadavg()[0])
        except Exception:
            load_1 = 0.0

        pressure = clamp01(max(cpu / 100.0, mem_pct / 100.0, min(1.0, load_1 / max(1.0, os.cpu_count() or 1))))
        return ResourceSnapshot(
            cpu_percent=cpu,
            memory_percent=mem_pct,
            memory_used_mb=used,
            memory_available_mb=avail,
            load_average_1m=load_1,
            pressure=pressure,
        )

    @property
    def high_pressure(self) -> bool:
        return self._last_snapshot.pressure >= self.high_pressure_threshold

    def status(self) -> Dict[str, Any]:
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
