"""Controlled full-weight self-training substrate.

This is intentionally small and CPU-only for CI and live-training coexistence,
but it performs real full-weight training: every parameter in the model is
updated by backpropagation, saved as an artifact, evaluated on hidden data, and
eligible for hot-swap promotion.
"""
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from core.promotion.gate import PromotionGate, ScoreEstimate
from core.runtime.atomic_writer import atomic_write_text
from core.runtime.hot_swap import HotSwapRegistry


Example = Tuple[Tuple[float, float], int]


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 7
    hidden_units: int = 4
    epochs: int = 1200
    learning_rate: float = 0.25
    train_size: int = 96
    hidden_eval_size: int = 96


@dataclass(frozen=True)
class FullWeightArtifact:
    artifact_id: str
    path: str
    train_accuracy: float
    hidden_accuracy: float
    baseline_hidden_accuracy: float
    promoted: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TinyMLP:
    """Two-layer MLP with all weights trainable."""

    def __init__(self, *, seed: int = 0, hidden_units: int = 4):
        rng = random.Random(seed)
        self.hidden_units = int(hidden_units)
        self.w1 = [[rng.uniform(-0.7, 0.7) for _ in range(2)] for _ in range(self.hidden_units)]
        self.b1 = [rng.uniform(-0.1, 0.1) for _ in range(self.hidden_units)]
        self.w2 = [rng.uniform(-0.7, 0.7) for _ in range(self.hidden_units)]
        self.b2 = rng.uniform(-0.1, 0.1)

    def predict_proba(self, x: Tuple[float, float]) -> float:
        hidden = [math.tanh(self.w1[i][0] * x[0] + self.w1[i][1] * x[1] + self.b1[i]) for i in range(self.hidden_units)]
        z = sum(self.w2[i] * hidden[i] for i in range(self.hidden_units)) + self.b2
        return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, z))))

    def predict(self, x: Tuple[float, float]) -> int:
        return int(self.predict_proba(x) >= 0.5)

    def train(self, examples: Sequence[Example], *, epochs: int, learning_rate: float) -> None:
        for _ in range(max(1, int(epochs))):
            for x, y in examples:
                hidden_raw = [self.w1[i][0] * x[0] + self.w1[i][1] * x[1] + self.b1[i] for i in range(self.hidden_units)]
                hidden = [math.tanh(v) for v in hidden_raw]
                z = sum(self.w2[i] * hidden[i] for i in range(self.hidden_units)) + self.b2
                p = 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, z))))
                dz = p - y
                old_w2 = list(self.w2)
                for i in range(self.hidden_units):
                    self.w2[i] -= learning_rate * dz * hidden[i]
                self.b2 -= learning_rate * dz
                for i in range(self.hidden_units):
                    dh = dz * old_w2[i] * (1.0 - hidden[i] * hidden[i])
                    self.w1[i][0] -= learning_rate * dh * x[0]
                    self.w1[i][1] -= learning_rate * dh * x[1]
                    self.b1[i] -= learning_rate * dh

    def accuracy(self, examples: Sequence[Example]) -> float:
        if not examples:
            return 0.0
        return sum(1 for x, y in examples if self.predict(x) == y) / len(examples)

    def to_dict(self) -> Dict[str, Any]:
        return {"hidden_units": self.hidden_units, "w1": self.w1, "b1": self.b1, "w2": self.w2, "b2": self.b2}

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TinyMLP":
        obj = cls(seed=0, hidden_units=int(payload["hidden_units"]))
        obj.w1 = [[float(v) for v in row] for row in payload["w1"]]
        obj.b1 = [float(v) for v in payload["b1"]]
        obj.w2 = [float(v) for v in payload["w2"]]
        obj.b2 = float(payload["b2"])
        return obj


class FullWeightTrainingEngine:
    """Train, evaluate, persist, and optionally promote a full-weight model."""

    def __init__(self, artifact_dir: Path | str):
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def run(self, config: Optional[TrainingConfig] = None, *, promote: bool = True) -> FullWeightArtifact:
        cfg = config or TrainingConfig()
        train = make_xor_dataset(cfg.seed, cfg.train_size)
        hidden = make_xor_dataset(cfg.seed + 10_000, cfg.hidden_eval_size)

        baseline = LinearThresholdBaseline()
        baseline_hidden = baseline.accuracy(hidden)

        model = TinyMLP(seed=cfg.seed, hidden_units=cfg.hidden_units)
        model.train(train, epochs=cfg.epochs, learning_rate=cfg.learning_rate)
        train_acc = model.accuracy(train)
        hidden_acc = model.accuracy(hidden)

        gate = PromotionGate(critical_metrics=["hidden_accuracy"], delta=0.05, emit_receipts=False)
        gate.compare({"hidden_accuracy": ScoreEstimate(baseline_hidden, stderr=0.0, n=len(hidden))})
        decision = gate.compare({"hidden_accuracy": ScoreEstimate(hidden_acc, stderr=0.0, n=len(hidden))})
        promoted = bool(promote and decision.accepted)

        artifact_id = f"full-weight-{int(time.time() * 1000)}"
        path = self.artifact_dir / f"{artifact_id}.json"
        payload = {
            "artifact_id": artifact_id,
            "config": asdict(cfg),
            "model": model.to_dict(),
            "train_accuracy": train_acc,
            "hidden_accuracy": hidden_acc,
            "baseline_hidden_accuracy": baseline_hidden,
            "promotion_decision": decision.to_dict(),
        }
        atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")

        if promoted:
            registry = HotSwapRegistry()
            registry.register("tiny_full_weight_model", baseline)
            ticket = registry.prepare("tiny_full_weight_model", model, validator=lambda candidate: candidate.accuracy(hidden) >= hidden_acc)
            result = registry.promote(ticket.ticket_id)
            promoted = bool(result.ok)

        return FullWeightArtifact(
            artifact_id=artifact_id,
            path=str(path),
            train_accuracy=train_acc,
            hidden_accuracy=hidden_acc,
            baseline_hidden_accuracy=baseline_hidden,
            promoted=promoted,
            metadata={"task": "continuous_xor", "all_weights_updated": True, "model_family": "TinyMLP"},
        )


class LinearThresholdBaseline:
    def predict(self, x: Tuple[float, float]) -> int:
        return int((x[0] + x[1]) >= 1.0)

    def accuracy(self, examples: Sequence[Example]) -> float:
        if not examples:
            return 0.0
        return sum(1 for x, y in examples if self.predict(x) == y) / len(examples)


def make_xor_dataset(seed: int, n: int) -> List[Example]:
    rng = random.Random(seed)
    out: List[Example] = []
    for _ in range(max(1, int(n))):
        x0 = rng.random()
        x1 = rng.random()
        label = int((x0 > 0.5) != (x1 > 0.5))
        out.append(((x0, x1), label))
    return out


__all__ = [
    "FullWeightArtifact",
    "FullWeightTrainingEngine",
    "LinearThresholdBaseline",
    "TinyMLP",
    "TrainingConfig",
    "make_xor_dataset",
]
