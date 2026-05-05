"""Benchmark ablation configuration."""
from __future__ import annotations

from dataclasses import dataclass, field

ABLATION_MODES = {
    "full",
    "prompt_only",
    "random_legal",
    "no_belief_graph",
    "no_modal_manager",
    "no_homeostasis",
    "no_simulator",
    "no_procedural_memory",
    "no_outcome_learning",
    "no_action_gateway_dry_run_only",
}


@dataclass
class AblationConfig:
    mode: str
    disabled_components: set[str] = field(default_factory=set)
    dry_run_only: bool = False

    def __post_init__(self) -> None:
        if self.mode not in ABLATION_MODES:
            raise ValueError(f"unknown_ablation_mode:{self.mode}")
        if self.mode == "no_action_gateway_dry_run_only":
            self.dry_run_only = True


__all__ = ["ABLATION_MODES", "AblationConfig"]
