"""Minimax weakest-domain curriculum: train the gap, protect the strengths.

Average-reward curricula create specialists: math progress conceals social
stagnation and the aggregate still climbs. Wholesale capability needs the
minimax objective — optimize the WORST domain relative to a matched frontier
reference — with replay protecting what is already strong:

    P(d) ∝ (1 − S_d(Aura) / S_d(reference))^γ        (gap term)
    every domain ⩾ a replay floor                     (anti-forgetting term)

Honesty rules:
- A domain with too few measurements gets an explicit EXPLORATION share and
  is never treated as strong by default — absence of evidence is not
  competence.
- Reference scores of zero (or no reference) make a gap uncomputable; the
  domain is receipted as such and gets the exploration share, not a made-up
  gap of 1.0.
- Allocation is deterministic (largest-remainder rounding, stable tie
  order), so the same measurements always produce the same curriculum and
  the receipt shows the whole computation.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("Aura.Learning.MinimaxCurriculum")

MINIMAX_CURRICULUM_SCHEMA = "aura.minimax_curriculum.v1"

MIN_MEASUREMENTS = 10
MAX_DOMAINS = 32


@dataclass(frozen=True)
class DomainMeasurement:
    """Matched scores for one capability domain."""

    domain: str
    aura_score: float  # success fraction in [0, 1]
    reference_score: float | None  # matched frontier reference; None ⇒ unknown
    n: int  # measurements behind aura_score

    def validated(self) -> DomainMeasurement:
        if not self.domain.strip():
            raise ValueError("domain measurement requires a domain name")
        if (
            isinstance(self.aura_score, bool)
            or not isinstance(self.aura_score, (int, float))
            or not math.isfinite(float(self.aura_score))
            or not 0.0 <= float(self.aura_score) <= 1.0
        ):
            raise ValueError("aura_score must be a finite fraction in [0, 1]")
        if self.reference_score is not None and (
            isinstance(self.reference_score, bool)
            or not isinstance(self.reference_score, (int, float))
            or not math.isfinite(float(self.reference_score))
            or not 0.0 <= float(self.reference_score) <= 1.0
        ):
            raise ValueError("reference_score must be a finite fraction or None")
        if type(self.n) is not int or self.n < 0:
            raise ValueError("n must be a non-negative integer")
        return self


class MinimaxCurriculumAllocator:
    """Deterministic gap-weighted allocation with replay + exploration floors."""

    def __init__(
        self,
        *,
        gamma: float = 2.0,
        replay_floor: float = 0.10,
        exploration_share: float = 0.15,
        min_measurements: int = MIN_MEASUREMENTS,
    ) -> None:
        if not (isinstance(gamma, (int, float)) and 0.5 <= float(gamma) <= 8.0):
            raise ValueError("gamma must be inside [0.5, 8]")
        if not 0.0 <= float(replay_floor) <= 0.5:
            raise ValueError("replay_floor must be inside [0, 0.5]")
        if not 0.0 <= float(exploration_share) <= 0.5:
            raise ValueError("exploration_share must be inside [0, 0.5]")
        if type(min_measurements) is not int or min_measurements < 1:
            raise ValueError("min_measurements must be a positive integer")
        self.gamma = float(gamma)
        self.replay_floor = float(replay_floor)
        self.exploration_share = float(exploration_share)
        self.min_measurements = min_measurements

    def allocate(
        self,
        measurements: list[DomainMeasurement],
        *,
        budget_items: int,
    ) -> dict[str, Any]:
        """Split ``budget_items`` training slots across domains, receipted."""
        if type(budget_items) is not int or budget_items < 1:
            raise ValueError("budget_items must be a positive integer")
        rows = [m.validated() for m in measurements]
        if not rows:
            raise ValueError("allocation requires at least one domain")
        if len(rows) > MAX_DOMAINS:
            raise ValueError(f"allocation supports at most {MAX_DOMAINS} domains")
        names = [row.domain for row in rows]
        if len(set(names)) != len(names):
            raise ValueError("domain names must be unique")

        gap_weights: dict[str, float] = {}
        exploration: set[str] = set()
        gap_notes: dict[str, str] = {}
        for row in rows:
            if row.n < self.min_measurements:
                exploration.add(row.domain)
                gap_notes[row.domain] = "underpowered_measurement"
                continue
            if row.reference_score is None or row.reference_score <= 0.0:
                exploration.add(row.domain)
                gap_notes[row.domain] = "no_computable_reference_gap"
                continue
            ratio = min(1.0, float(row.aura_score) / float(row.reference_score))
            gap_weights[row.domain] = (1.0 - ratio) ** self.gamma
            gap_notes[row.domain] = f"gap_ratio={1.0 - ratio:.4f}"

        # Shares: replay floor for everyone, exploration pool for the
        # unmeasurable, the rest split by gap weight.
        shares: dict[str, float] = {name: 0.0 for name in names}
        replay_each = self.replay_floor / len(names)
        for name in names:
            shares[name] += replay_each
        remaining = 1.0 - self.replay_floor
        exploration_pool = self.exploration_share * remaining if exploration else 0.0
        if exploration:
            per_exploration = exploration_pool / len(exploration)
            for name in exploration:
                shares[name] += per_exploration
        gap_pool = remaining - exploration_pool
        total_gap = sum(gap_weights.values())
        if total_gap > 0.0:
            for name, weight in gap_weights.items():
                shares[name] += gap_pool * (weight / total_gap)
        else:
            # Nothing measurably behind: the gap pool becomes uniform replay,
            # which is exactly what "protect the strengths" means at parity.
            for name in names:
                shares[name] += gap_pool / len(names)

        counts = _largest_remainder(shares, budget_items)
        weakest = (
            min(
                (row for row in rows if row.domain in gap_weights),
                key=lambda row: (
                    float(row.aura_score) / float(row.reference_score),
                    row.domain,
                ),
            ).domain
            if gap_weights
            else None
        )
        return {
            "schema": MINIMAX_CURRICULUM_SCHEMA,
            "budget_items": budget_items,
            "gamma": self.gamma,
            "replay_floor": self.replay_floor,
            "exploration_share": self.exploration_share,
            "weakest_domain": weakest,
            "exploration_domains": sorted(exploration),
            "gap_notes": gap_notes,
            "shares": {name: round(share, 6) for name, share in shares.items()},
            "counts": counts,
        }


def _largest_remainder(shares: dict[str, float], total: int) -> dict[str, int]:
    """Deterministic apportionment: floors first, remainders by size then name."""
    floors = {name: int(math.floor(share * total)) for name, share in shares.items()}
    assigned = sum(floors.values())
    remainders = sorted(
        shares,
        key=lambda name: (-(shares[name] * total - floors[name]), name),
    )
    for name in remainders[: total - assigned]:
        floors[name] += 1
    return floors


__all__ = [
    "DomainMeasurement",
    "MAX_DOMAINS",
    "MIN_MEASUREMENTS",
    "MINIMAX_CURRICULUM_SCHEMA",
    "MinimaxCurriculumAllocator",
]
