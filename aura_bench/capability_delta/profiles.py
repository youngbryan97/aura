"""Ablation profiles — what gets disabled in each evaluation run.

A capability-delta run scores the same task suite under several
profiles.  ``full`` enables every Aura subsystem; ``base_llm_only``
disables everything around the base LLM (no memory, no homeostasis,
no global workspace, no Will, no affect, no curriculum) so we can
measure how much each subsystem contributes.

Profiles are pure data — adapters interpret them by deciding which
services to register / shim out before running a task.  The set is
closed and named so reports stay comparable across runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Tuple


@dataclass(frozen=True)
class AblationProfile:
    """A named set of subsystem flags used to gate a benchmark run."""

    name: str
    description: str
    enabled_subsystems: FrozenSet[str]

    def disabled_subsystems(self, all_subsystems: FrozenSet[str]) -> FrozenSet[str]:
        return frozenset(all_subsystems - self.enabled_subsystems)

    def enables(self, subsystem: str) -> bool:
        return subsystem in self.enabled_subsystems


# All subsystems an ablation can toggle.  New subsystems must be added
# here to participate in delta scoring.
KNOWN_SUBSYSTEMS: FrozenSet[str] = frozenset(
    {
        "memory",
        "homeostasis",
        "global_workspace",
        "will",
        "affect",
        "curriculum",
        "tools",
        "planner",
    }
)


def _all_enabled() -> FrozenSet[str]:
    return KNOWN_SUBSYSTEMS


ABLATION_PROFILES: Tuple[AblationProfile, ...] = (
    AblationProfile(
        name="full",
        description="All subsystems enabled — Aura at full capacity.",
        enabled_subsystems=_all_enabled(),
    ),
    AblationProfile(
        name="no_memory",
        description="Disable memory facade — measures memory contribution.",
        enabled_subsystems=_all_enabled() - frozenset({"memory"}),
    ),
    AblationProfile(
        name="no_homeostasis",
        description="Disable homeostasis loop — measures regulatory contribution.",
        enabled_subsystems=_all_enabled() - frozenset({"homeostasis"}),
    ),
    AblationProfile(
        name="no_global_workspace",
        description="Disable Global Workspace Theory bus — measures GWT contribution.",
        enabled_subsystems=_all_enabled() - frozenset({"global_workspace"}),
    ),
    AblationProfile(
        name="no_will",
        description="Disable Unified Will / Authority gateway.",
        enabled_subsystems=_all_enabled() - frozenset({"will"}),
    ),
    AblationProfile(
        name="no_affect",
        description="Disable affect/substrate — measures emotional regulation contribution.",
        enabled_subsystems=_all_enabled() - frozenset({"affect"}),
    ),
    AblationProfile(
        name="base_llm_only",
        description="Strip every Aura subsystem; base LLM with naive prompting.",
        enabled_subsystems=frozenset(),
    ),
)


_BY_NAME: Dict[str, AblationProfile] = {p.name: p for p in ABLATION_PROFILES}


def profile_by_name(name: str) -> AblationProfile:
    if name not in _BY_NAME:
        raise KeyError(
            f"unknown ablation profile {name!r}; "
            f"valid: {sorted(_BY_NAME.keys())}"
        )
    return _BY_NAME[name]
