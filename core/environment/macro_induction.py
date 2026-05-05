"""Mine repeated successful traces into macro candidates."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .command import ActionIntent


@dataclass
class MacroCandidate:
    name: str
    environment_family: str
    trigger_signature: str
    step_template: list[ActionIntent]
    success_rate: float
    mean_cost: float
    risk_score: float
    examples: list[str] = field(default_factory=list)
    validated: bool = False


class MacroInducer:
    def mine(self, traces: list[list[str]], *, environment_family: str, trigger_signature: str) -> list[MacroCandidate]:
        windows: Counter[tuple[str, ...]] = Counter()
        for trace in traces:
            for size in range(2, min(5, len(trace)) + 1):
                for idx in range(0, len(trace) - size + 1):
                    windows[tuple(trace[idx : idx + size])] += 1
        candidates: list[MacroCandidate] = []
        for sequence, count in windows.items():
            if count < 2:
                continue
            candidates.append(
                MacroCandidate(
                    name="macro_" + "_".join(sequence[:3]),
                    environment_family=environment_family,
                    trigger_signature=trigger_signature,
                    step_template=[ActionIntent(name=step) for step in sequence],
                    success_rate=1.0,
                    mean_cost=float(len(sequence)),
                    risk_score=0.1,
                    examples=[" -> ".join(sequence)],
                )
            )
        return candidates


__all__ = ["MacroCandidate", "MacroInducer"]
