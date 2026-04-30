"""External-usefulness validation for STDP plasticity.

This experiment distinguishes useful learning from endogenous drift.  Four
groups are trained with identical mechanics:

A: external environment signal
B: self-generated signal
C: frozen matrix
D: shuffled environment control

The pass condition is external training beating all controls on held-out
one-step prediction while staying within the instability budget.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.consciousness.stdp_learning import STDPLearningEngine
from core.runtime.atomic_writer import atomic_write_text


@dataclass(frozen=True)
class STDPGroupResult:
    group: str
    heldout_mse: float
    instability: float
    weight_norm: float
    updates: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "heldout_mse": round(self.heldout_mse, 6),
            "instability": round(self.instability, 6),
            "weight_norm": round(self.weight_norm, 6),
            "updates": self.updates,
        }


@dataclass(frozen=True)
class STDPExternalValidationReport:
    generated_at: float
    seed: int
    steps: int
    passed: bool
    groups: tuple[STDPGroupResult, ...]
    margins: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "seed": self.seed,
            "steps": self.steps,
            "passed": self.passed,
            "groups": [group.to_dict() for group in self.groups],
            "margins": {k: round(v, 6) for k, v in self.margins.items()},
        }


class STDPExternalValidator:
    def __init__(self, n_neurons: int = 16, seed: int = 7) -> None:
        self.n = n_neurons
        self.seed = seed

    def run(self, steps: int = 96) -> STDPExternalValidationReport:
        rng = np.random.default_rng(self.seed)
        env = self._environment_sequence(steps * 2 + 4, rng)
        groups = (
            self._run_group("external_environment", env[:steps], env[steps:], train_mode="external"),
            self._run_group("self_generated", env[:steps], env[steps:], train_mode="self"),
            self._run_group("frozen_matrix", env[:steps], env[steps:], train_mode="frozen"),
            self._run_group("shuffled_environment", env[:steps], env[steps:], train_mode="shuffled"),
        )
        by_name = {group.group: group for group in groups}
        external = by_name["external_environment"]
        margins = {
            name: result.heldout_mse - external.heldout_mse
            for name, result in by_name.items()
            if name != "external_environment"
        }
        max_control_instability = max(result.instability for name, result in by_name.items() if name != "external_environment")
        passed = (
            all(margin > 0.015 for margin in margins.values())
            and external.instability <= max(0.20, max_control_instability + 0.08)
        )
        return STDPExternalValidationReport(
            generated_at=time.time(),
            seed=self.seed,
            steps=steps,
            passed=passed,
            groups=groups,
            margins=margins,
        )

    def _run_group(
        self,
        name: str,
        train_seq: np.ndarray,
        heldout_seq: np.ndarray,
        *,
        train_mode: str,
    ) -> STDPGroupResult:
        engine = STDPLearningEngine(n_neurons=self.n)
        W = np.zeros((self.n, self.n), dtype=np.float32)
        previous_W = W.copy()
        perm = np.random.default_rng(self.seed + 200).permutation(len(train_seq))
        shuffled = train_seq[perm]
        self_generated = self._self_generated_sequence(len(train_seq))

        if train_mode == "frozen":
            trained_W = W
            updates = 0
        else:
            for idx in range(len(train_seq) - 1):
                current = train_seq[idx]
                target = train_seq[idx + 1]
                if train_mode == "shuffled":
                    current = shuffled[idx]
                    target = shuffled[idx + 1]
                elif train_mode == "self":
                    current = self_generated[idx]
                    target = self_generated[idx + 1]
                pred = self._sigmoid(current @ W)
                error = float(np.mean((pred - target) ** 2))
                engine.record_spikes(current, t=idx * 20.0)
                dw = engine.deliver_reward(surprise=min(1.0, error * 4.0), prediction_error=error)
                hebbian = np.outer(current, target - pred).astype(np.float32) * 0.018
                W = engine.apply_to_connectivity(W, (dw * 18.0) + hebbian)
            trained_W = W
            updates = engine.get_status()["total_updates"]

        mse = self._heldout_mse(trained_W, heldout_seq)
        instability = float(np.linalg.norm(trained_W - previous_W) / max(1.0, self.n))
        return STDPGroupResult(
            group=name,
            heldout_mse=mse,
            instability=instability,
            weight_norm=float(np.linalg.norm(trained_W)),
            updates=int(updates),
        )

    def _heldout_mse(self, W: np.ndarray, heldout_seq: np.ndarray) -> float:
        errors: list[float] = []
        for idx in range(len(heldout_seq) - 1):
            pred = self._sigmoid(heldout_seq[idx] @ W)
            errors.append(float(np.mean((pred - heldout_seq[idx + 1]) ** 2)))
        return float(np.mean(errors))

    def _environment_sequence(self, length: int, rng: np.random.Generator) -> np.ndarray:
        seq = np.zeros((length, self.n), dtype=np.float32)
        phase = 0.0
        for t in range(length):
            phase += 0.22
            base = 0.5 + 0.45 * np.sin(phase + np.arange(self.n) * 0.19)
            load = 0.25 + 0.70 * ((t // 8) % 2)
            tool = 0.85 if (t % 11) in {3, 4, 5} else 0.15
            retrieval = 0.75 if np.sin(phase * 0.5) > 0 else 0.25
            base[0] = load
            base[1] = tool
            base[2] = retrieval
            base[3] = 1.0 - abs(load - retrieval)
            noise = rng.normal(0.0, 0.025, size=self.n)
            seq[t] = np.clip(base + noise, 0.0, 1.0)
        return seq

    def _self_generated_sequence(self, length: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed + 100)
        seq = rng.uniform(0.25, 0.75, size=(length, self.n)).astype(np.float32)
        for t in range(1, length):
            seq[t] = np.clip((0.86 * seq[t - 1]) + (0.14 * seq[t]), 0.0, 1.0)
        return seq

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-values))


def run_and_write(path: str | Path, *, steps: int = 96, seed: int = 7) -> STDPExternalValidationReport:
    report = STDPExternalValidator(seed=seed).run(steps=steps)
    atomic_write_text(Path(path), json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return report


__all__ = ["STDPExternalValidator", "STDPExternalValidationReport", "STDPGroupResult", "run_and_write"]
