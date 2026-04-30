"""Sandboxed meta-learning over plasticity rule parameters."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from core.consciousness.stdp_external_validation import STDPExternalValidator


@dataclass(frozen=True)
class PlasticityVariant:
    name: str
    learning_scale: float
    stability_penalty: float


@dataclass(frozen=True)
class PlasticityTournamentResult:
    winner: PlasticityVariant
    scores: dict[str, float]
    validation: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "winner": self.winner.__dict__,
            "scores": self.scores,
            "validation": self.validation,
        }


class MetaPlasticityTournament:
    """Evaluates plasticity variants in sandboxed validation, never live math."""

    DEFAULT_VARIANTS = (
        PlasticityVariant("conservative", 0.75, 0.15),
        PlasticityVariant("baseline", 1.0, 0.12),
        PlasticityVariant("adaptive", 1.25, 0.18),
    )

    def run(self, *, steps: int = 64, seed: int = 11) -> PlasticityTournamentResult:
        validator = STDPExternalValidator(seed=seed)
        base_report = validator.run(steps=steps)
        external = next(group for group in base_report.groups if group.group == "external_environment")
        scores: dict[str, float] = {}
        for variant in self.DEFAULT_VARIANTS:
            usefulness = 1.0 / max(1e-6, external.heldout_mse / variant.learning_scale)
            stability = max(0.0, 1.0 - external.instability - variant.stability_penalty)
            scores[variant.name] = float(usefulness * 0.7 + stability * 0.3)
        winner = max(self.DEFAULT_VARIANTS, key=lambda v: scores[v.name])
        return PlasticityTournamentResult(winner=winner, scores=scores, validation=base_report.to_dict())


__all__ = ["PlasticityVariant", "PlasticityTournamentResult", "MetaPlasticityTournament"]
