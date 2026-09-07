"""Which faculties an ablation can remove without booting the whole mind.

The matched-substrate protocol generates by calling the model directly, and
reports two of its three ablation arms as NOT_MEASURED because the channels
they remove act somewhere the call never goes. That reading is right and it
was reached by running the arms and finding they sampled identically, which is
an expensive way to learn something the source already knows.

A lesion arrives three ways and the first version of this knew one of them.
``apply_channel`` is the explicit form; ``register_flag_lesion`` binds a
channel to a flag the faculty itself reads; ``@lesionable`` binds it to a
class's own ``lesion()``. Counting only the first reported three channels as
declared and inert when they are lesioned perfectly well by the other two —
which is the same mistake as measuring a faculty by the one path you happened
to look at.

Where the site sits decides what can measure it: a channel applied inside
``inference_gate`` bites any generation that goes through the gate, and one
inside ``cognitive_engine`` bites only a turn that goes through the engine. So
"which arms can this harness build" is answered by reading where the sites
are, before spending a thousand generations to find out.

Two numbers come out of that and they are different questions. How many
channels are declared is a fact about the apparatus. How many a given harness
can actually move is a fact about the harness, and it is the one that says
whether an ablation study is worth running at all.

Nothing here decides anything. It reports where each faculty acts, so a
protocol can say which arms it is entitled to claim.
"""
from __future__ import annotations

import ast
import functools
import logging
import pathlib
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("Aura.WhichLesionsADirectCallCanBite")

__all__ = [
    "AChannelSite",
    "WHAT_A_HARNESS_REACHES",
    "where_each_channel_acts",
    "what_a_direct_call_can_bite",
    "how_the_lesions_are_reachable",
]

#: What each kind of harness gets to execute, by the module a channel is
#: applied in. A harness that calls the model itself never enters the engine,
#: so a channel applied there cannot move for it however carefully the arm is
#: built.
WHAT_A_HARNESS_REACHES: dict[str, tuple[str, ...]] = {
    "a direct model call": ("core/brain/inference_gate.py",),
    "a turn through the gate": ("core/brain/inference_gate.py",),
    "a turn through the engine": (
        "core/brain/inference_gate.py",
        "core/brain/cognitive_engine.py",
        "core/consciousness/qualia_synthesizer.py",
        "core/affect/affective_circumplex.py",
        "core/being/affective_valence.py",
    ),
}

#: The three ways a lesion is bound to a channel. All of them count as the
#: channel being lesionable somewhere.
HOW_A_LESION_IS_BOUND: tuple[str, ...] = (
    "apply_channel",
    "register_flag_lesion",
    "lesionable",
)


@dataclass(frozen=True, slots=True)
class AChannelSite:
    """One lesion channel and where it is actually applied."""

    channel: str
    applied_in: tuple[str, ...]

    def reachable_by(self, harness: str) -> bool:
        allowed = WHAT_A_HARNESS_REACHES.get(harness, ())
        return any(one.startswith(allowed) for one in self.applied_in) if allowed else False

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "applied_in": list(self.applied_in),
            "reachable_by": sorted(
                harness
                for harness in WHAT_A_HARNESS_REACHES
                if self.reachable_by(harness)
            ),
        }


def _constant_names(repo: pathlib.Path) -> dict[str, str]:
    """The channel constants, by the name they are referred to as."""
    found: dict[str, str] = {}
    path = repo / "core" / "verify" / "influence_channels.py"
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except (OSError, SyntaxError, ValueError):
        return found
    for node in tree.body:
        target = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target, value = node.target.id, node.value
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(
            node.targets[0], ast.Name
        ):
            target, value = node.targets[0].id, node.value
        else:
            continue
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            found[target] = value.value
    return found


@functools.lru_cache(maxsize=2)
def where_each_channel_acts(repo: str = ".") -> tuple[AChannelSite, ...]:
    """Every declared channel, and the files that apply it.

    A channel with no application site is declared and inert: nothing anywhere
    lesions it, so an arm removing it is identical to intact whatever harness
    runs the arm.
    """
    root = pathlib.Path(repo)
    names = _constant_names(root)
    sites: dict[str, set[str]] = {value: set() for value in names.values()}
    for path in (root / "core").rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except (OSError, SyntaxError, ValueError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            called = getattr(node.func, "id", getattr(node.func, "attr", ""))
            if called not in HOW_A_LESION_IS_BOUND:
                continue
            asked = node.args[0]
            key = ""
            if isinstance(asked, ast.Attribute):
                key = names.get(asked.attr, "")
            elif isinstance(asked, ast.Name):
                key = names.get(asked.id, "")
            elif isinstance(asked, ast.Constant) and isinstance(asked.value, str):
                key = asked.value
            if key:
                sites.setdefault(key, set()).add(str(path.relative_to(root)))
    return tuple(
        AChannelSite(channel=name, applied_in=tuple(sorted(where)))
        for name, where in sorted(sites.items())
    )


def what_a_direct_call_can_bite(repo: str = ".") -> tuple[str, ...]:
    """Channels a harness that calls the model itself can actually move."""
    return tuple(
        one.channel
        for one in where_each_channel_acts(repo)
        if one.reachable_by("a direct model call")
    )


def how_the_lesions_are_reachable(repo: str = ".") -> dict[str, Any]:
    """For the health report: declared, applied, and reachable by what."""
    every = where_each_channel_acts(repo)
    inert = [one.channel for one in every if not one.applied_in]
    return {
        "schema": "aura.lesions.reachability.v1",
        "declared": len(every),
        "applied_somewhere": sum(1 for one in every if one.applied_in),
        # Declared and never lesioned: an arm that removes one of these is
        # identical to intact whatever runs it.
        "declared_and_inert": inert,
        "reachable_by": {
            harness: sorted(
                one.channel for one in every if one.reachable_by(harness)
            )
            for harness in WHAT_A_HARNESS_REACHES
        },
        "each": [one.to_dict() for one in every],
    }
