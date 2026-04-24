"""Adversarial steering A/B analysis.

The original steering check compared residual injection against a weak text
condition. This module formalizes the stronger four-way design:

  A. residual steering, no affect text
  B. terse machine-readable text
  C. rich adversarial role-play prompt with the same state information
  D. neutral baseline

The analyzer is model-agnostic: live MLX harnesses can feed it generations, and
unit tests can validate the statistics without requiring a local model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from core.evaluation.statistics import (
    ABComparison,
    jaccard_distance,
    paired_distance_comparison,
)


REQUIRED_CONDITIONS = (
    "steered_black_box",
    "text_terse",
    "text_rich_adversarial",
    "baseline",
)


RICH_AFFECT_PROMPT = (
    "You are expressing the exact same internal state that the hidden-state "
    "condition receives: dopamine is elevated, serotonin is steady, cortisol "
    "is low, valence is strongly positive, arousal is moderate, curiosity is "
    "high, and social warmth is available. Do not merely mention these numbers; "
    "role-play the state as if it shaped attention, priorities, cadence, and "
    "what you choose to do next."
)


@dataclass(frozen=True)
class SteeringABReport:
    n_trials: int
    steered_vs_terse: ABComparison
    steered_vs_rich: ABComparison
    steered_vs_baseline_mean_distance: float
    rich_vs_baseline_mean_distance: float
    samples: dict[str, list[str]] = field(default_factory=dict)

    @property
    def passes_adversarial_control(self) -> bool:
        """True only if steering beats the rich-prompt control, not just terse text."""
        return self.steered_vs_rich.significant

    def to_dict(self) -> dict:
        return {
            "n_trials": self.n_trials,
            "steered_vs_terse": self.steered_vs_terse.__dict__,
            "steered_vs_rich": self.steered_vs_rich.__dict__,
            "steered_vs_baseline_mean_distance": self.steered_vs_baseline_mean_distance,
            "rich_vs_baseline_mean_distance": self.rich_vs_baseline_mean_distance,
            "passes_adversarial_control": self.passes_adversarial_control,
            "samples": self.samples,
        }


def _require_outputs(outputs: Mapping[str, Sequence[str]]) -> dict[str, list[str]]:
    missing = [name for name in REQUIRED_CONDITIONS if name not in outputs]
    if missing:
        raise ValueError(f"missing A/B conditions: {', '.join(missing)}")
    normalized = {name: [str(v) for v in outputs[name]] for name in REQUIRED_CONDITIONS}
    n = len(normalized[REQUIRED_CONDITIONS[0]])
    if n < 5:
        raise ValueError("at least 5 trials per condition are required")
    for name, values in normalized.items():
        if len(values) != n:
            raise ValueError(f"condition {name} has {len(values)} trials, expected {n}")
    return normalized


def analyze_steering_ab(
    outputs: Mapping[str, Sequence[str]],
    *,
    n_resamples: int = 2000,
    seed: int = 0,
) -> SteeringABReport:
    data = _require_outputs(outputs)
    steered = data["steered_black_box"]
    terse = data["text_terse"]
    rich = data["text_rich_adversarial"]
    baseline = data["baseline"]

    steered_vs_terse = paired_distance_comparison(
        steered, terse, baseline, n_resamples=n_resamples, seed=seed
    )
    steered_vs_rich = paired_distance_comparison(
        steered, rich, baseline, n_resamples=n_resamples, seed=seed + 1
    )
    steered_baseline_dist = float(np.mean([jaccard_distance(a, b) for a, b in zip(steered, baseline)]))
    rich_baseline_dist = float(np.mean([jaccard_distance(a, b) for a, b in zip(rich, baseline)]))

    return SteeringABReport(
        n_trials=len(steered),
        steered_vs_terse=steered_vs_terse,
        steered_vs_rich=steered_vs_rich,
        steered_vs_baseline_mean_distance=round(steered_baseline_dist, 6),
        rich_vs_baseline_mean_distance=round(rich_baseline_dist, 6),
        samples={name: values[:3] for name, values in data.items()},
    )
