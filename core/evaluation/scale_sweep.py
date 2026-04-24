"""Lightweight scale-sweep utilities.

This is not a substitute for full IIT on the deployed system.  It gives the
repo a repeatable, bounded artifact for the scale-generalization question:
when we increase substrate width, do simple integration proxies and hardware
feasibility collapse, and where do they collapse?
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from core.evaluation.hardware_reality import HardwareRealityAuditor, MachineProfile
from core.evaluation.statistics import mutual_information_discrete


@dataclass(frozen=True)
class ScaleSweepPoint:
    nodes: int
    proxy_mi: float
    shuffled_proxy_mi: float
    proxy_delta: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "nodes": self.nodes,
            "proxy_mi": round(self.proxy_mi, 6),
            "shuffled_proxy_mi": round(self.shuffled_proxy_mi, 6),
            "proxy_delta": round(self.proxy_delta, 6),
        }


def run_integration_proxy_sweep(
    sizes: Iterable[int] = (8, 32, 128, 512),
    *,
    steps: int = 256,
    seed: int = 123,
) -> list[ScaleSweepPoint]:
    rng = np.random.default_rng(seed)
    points: list[ScaleSweepPoint] = []
    for nodes in sizes:
        states = _coupled_binary_history(nodes, steps=steps, rng=rng)
        shuffled = states.copy()
        for col in range(shuffled.shape[1]):
            rng.shuffle(shuffled[:, col])
        proxy = _mean_neighbor_mi(states)
        shuffled_proxy = _mean_neighbor_mi(shuffled)
        points.append(
            ScaleSweepPoint(
                nodes=nodes,
                proxy_mi=proxy,
                shuffled_proxy_mi=shuffled_proxy,
                proxy_delta=proxy - shuffled_proxy,
            )
        )
    return points


def hardware_scale_table(machine: MachineProfile) -> list[dict[str, object]]:
    return [verdict.as_dict() for verdict in HardwareRealityAuditor(machine).evaluate_all()]


def _coupled_binary_history(nodes: int, *, steps: int, rng: np.random.Generator) -> np.ndarray:
    state = rng.integers(0, 2, size=nodes, dtype=np.int8)
    rows = []
    for _ in range(steps):
        rows.append(state.copy())
        left = np.roll(state, 1)
        right = np.roll(state, -1)
        noise = rng.random(nodes) < 0.04
        state = ((left ^ right) | (state & left)).astype(np.int8)
        state = np.where(noise, 1 - state, state).astype(np.int8)
    return np.array(rows, dtype=np.int8)


def _mean_neighbor_mi(states: np.ndarray) -> float:
    if states.shape[1] < 2:
        return 0.0
    stride = max(1, states.shape[1] // 32)
    values = []
    for idx in range(0, states.shape[1], stride):
        values.append(
            mutual_information_discrete(
                states[:-1, idx].tolist(),
                states[1:, (idx + 1) % states.shape[1]].tolist(),
            )
        )
    return float(np.mean(values)) if values else 0.0

