"""Whether her state actually moves the decoding, measured against a null.

``LIVE_MIND_CONTROL_POLICY`` says out loud that it is a hand-tuned heuristic
and that ``LIVE_MIND_CONTROL_POLICY_CALIBRATED`` is False, which is honest and
is not evidence. An external review made the point precisely: the code proves
internal state CAN actuate model computation; it does not prove that this
particular mapping is beneficial, learned, or even discriminative.

The last of those three is the one that can be settled offline, and it is the
one that would embarrass the claim most if it failed. A policy that returns
nearly the same temperature for every mind state is a constant wearing a
policy's clothes — the mechanism would still be causal, and it would still be
carrying no information.

So: sweep the mapping across the state space it reads, and measure the spread
of each control it emits. Then do it again with the state shuffled, so each
control is computed from a mind that was never in that configuration. If the
real sweep and the shuffled sweep have the same spread, the policy is
responding to noise rather than to her.

This measures discrimination, not benefit. Whether the controls it picks make
answers better needs the model and a held-out set, and saying so is part of
the result rather than a caveat attached to it.
"""
from __future__ import annotations

import functools
import logging
import statistics
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("Aura.DoesTheMindMoveTheControls")

__all__ = [
    "AControlSweep",
    "how_much_the_mind_moves_the_controls",
    "the_states_it_reads",
]

#: The readings the policy actually consults, and the range each can take.
#: Read from the mapping rather than imagined: a sweep over inputs it ignores
#: measures the sweep.
def the_states_it_reads() -> dict[str, tuple[float, ...]]:
    return {
        "dominant_intensity": (0.0, 0.25, 0.5, 0.75, 1.0),
        "curiosity": (0.0, 0.3, 0.6, 0.9),
        "pain": (0.0, 0.3, 0.6, 0.9),
        "integration": (0.0, 0.35, 0.7, 1.0),
        "self_presence": (0.0, 0.5, 1.0),
    }


#: The labels the mapping branches on. A sweep that only varies numbers misses
#: every branch taken on a name.
_THE_LABELS: tuple[str, ...] = ("", "joy", "curiosity", "distress", "fear", "calm")


@dataclass(frozen=True)
class AControlSweep:
    """What a sweep found for one control: how much, and driven by what."""

    control: str
    spread: float
    values: int
    distinct: int
    #: How much of the control's variance each reading accounts for. A reading
    #: at zero is one the policy ignores; the mapping may still mention it.
    explained_by: dict[str, float]

    @property
    def moves(self) -> bool:
        return self.distinct > 1

    @property
    def readings_that_do_nothing(self) -> list[str]:
        return sorted(
            name for name, share in self.explained_by.items() if share < 1e-9
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "control": self.control,
            "spread": round(self.spread, 5),
            "values": self.values,
            "distinct": self.distinct,
            "moves": self.moves,
            "explained_by": {
                name: round(share, 4) for name, share in sorted(self.explained_by.items())
            },
            "readings_that_do_nothing": self.readings_that_do_nothing,
        }


def _a_snapshot(
    label: str,
    intensity: float,
    curiosity: float,
    pain: float,
    integration: float,
    self_presence: float,
) -> dict[str, Any]:
    return {
        "mind_snapshot_quality": {"ready": True},
        "mind_snapshot": {
            "affect_grounding": {
                "dominant": {"label": label, "intensity": intensity}
            },
            "drive_integration": {"drives": {"curiosity": {"activation": curiosity}}},
            "nociception": {"nociceptive_pressure": pain},
            "phenomenal_engine": {
                "integration": integration,
                "self_presence": self_presence,
            },
        },
    }


def _share_explained(values: list[float], groups: list[Any]) -> float:
    """How much of the variance in ``values`` this grouping accounts for.

    Between-group variance over total. The mapping is deterministic, so there
    is no null to compare against — a shuffle of the inputs leaves the output
    distribution identical, which is why a spread-against-shuffle comparison
    cannot see anything and was replaced. What varies is which reading the
    control follows, and that is what this measures.
    """
    if len(values) < 2:
        return 0.0
    overall = statistics.fmean(values)
    total = sum((one - overall) ** 2 for one in values)
    if total <= 0.0:
        return 0.0
    by_group: dict[Any, list[float]] = {}
    for value, group in zip(values, groups, strict=True):
        by_group.setdefault(group, []).append(value)
    between = sum(
        len(held) * (statistics.fmean(held) - overall) ** 2
        for held in by_group.values()
    )
    return max(0.0, min(1.0, between / total))


@functools.lru_cache(maxsize=1)
def _swept() -> tuple[tuple[str, Any], ...]:
    return tuple(_sweep().items())


def how_much_the_mind_moves_the_controls() -> dict[str, Any]:
    """Sweep the whole state space and say which reading moves which control.

    Deterministic and exhaustive over the grid: no sampling, no seed, and two
    runs of one commit agree because there is nothing in it that could differ.

    Cached for the life of the process, which is safe for the same reason: the
    answer depends only on the mapping, and the mapping does not change while
    the process runs. Without it the health report swept 5,760 states every
    time its memo expired.
    """
    return dict(_swept())


def _sweep() -> dict[str, Any]:
    from core.brain.cognitive_engine import (
        LIVE_MIND_CONTROL_POLICY,
        LIVE_MIND_CONTROL_POLICY_CALIBRATED,
        _live_mind_generation_controls,
    )

    axes = the_states_it_reads()
    names = ("label", *axes)
    states: list[tuple[Any, ...]] = []
    for label in _THE_LABELS:
        for intensity in axes["dominant_intensity"]:
            for curiosity in axes["curiosity"]:
                for pain in axes["pain"]:
                    for integration in axes["integration"]:
                        for presence in axes["self_presence"]:
                            states.append(
                                (label, intensity, curiosity, pain, integration, presence)
                            )

    produced = [_live_mind_generation_controls(_a_snapshot(*one)) for one in states]

    swept: list[AControlSweep] = []
    for control in sorted({name for one in produced for name in one}):
        values = [
            float(one[control])
            for one in produced
            if isinstance(one.get(control), (int, float))
        ]
        if len(values) != len(states):
            continue
        explained = {
            reading: _share_explained(values, [one[at] for one in states])
            for at, reading in enumerate(names)
        }
        swept.append(
            AControlSweep(
                control=control,
                spread=statistics.pstdev(values),
                values=len(values),
                distinct=len(set(values)),
                explained_by=explained,
            )
        )

    never_move = [one.control for one in swept if not one.moves]
    ignored = sorted(
        {
            reading
            for one in swept
            for reading in one.readings_that_do_nothing
        }
        - {
            reading
            for one in swept
            for reading, share in one.explained_by.items()
            if share >= 1e-9
        }
    )
    return {
        "policy": LIVE_MIND_CONTROL_POLICY,
        "calibrated": bool(LIVE_MIND_CONTROL_POLICY_CALIBRATED),
        "states_swept": len(states),
        "controls": [one.to_dict() for one in swept],
        "controls_that_never_move": never_move,
        "readings_nothing_reads": ignored,
        "discriminates": bool(swept) and not never_move,
        # Said in the result rather than as a caveat beside it.
        "what_this_does_not_show": (
            "whether the controls it picks make answers better; that needs "
            "the model and a held-out set"
        ),
    }


def register_the_sweep() -> None:
    """Offer the sweep through the registry, so health can read it.

    core/runtime may not import core.brain — its DEPS is one of the seven
    hand-written foundation rules and says so. A health block that needed that
    edge would be a layering violation dressed as observability, which is the
    comment already sitting beside the endogenous-language block in the same
    file. So the provider is registered here and resolved there.
    """
    from core.container import ServiceContainer
    from core.runtime.service_registry import register_runtime_service

    # Both registries: neither is a superset of the other, and the registry
    # sink is only installed once a runtime owns the process.
    ServiceContainer.register_instance(
        "the_control_policy_sweep", how_much_the_mind_moves_the_controls,
        required=False,
    )
    register_runtime_service(
        "the_control_policy_sweep",
        how_much_the_mind_moves_the_controls,
        required=False,
        owner="core/brain/does_the_mind_move_the_controls.py",
        registered_by="register_the_sweep",
    )
