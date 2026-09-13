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
    "THE_FOREGROUND_GUARDS",
    "WHAT_A_HARNESS_REACHES",
    "what_a_background_gate_call_can_bite",
    "what_the_probe_turn_can_bite",
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
    # The hourly influence campaign. It asks the gate with a background,
    # non-user-facing origin, so it is in the gate's file and not in the
    # gate's foreground branch — which is where the circumplex is applied.
    # Entered separately because the campaign spent three weeks rotating
    # through nine channels on the strength of "the site is in the gate".
    "a background call through the gate": ("core/brain/inference_gate.py",),
    # What the campaign uses now: one turn down the desktop quick-reply lane,
    # with a foreground origin. It passes both guards, so the sites inside the
    # clean-user-surface contract and the gate's foreground branch are both on
    # its path. Same reach as a turn through the engine, because it is one.
    "a turn through the probe lane": (
        "core/brain/inference_gate.py",
        "core/brain/cognitive_engine.py",
        "core/consciousness/qualia_synthesizer.py",
        "core/affect/affective_circumplex.py",
        "core/being/affective_valence.py",
    ),
}

#: What a test has to mention for the branch under it to be foreground-only.
#: Narrow and literal on purpose: this is read off the source, so it has to
#: name the actual guards rather than guess at intent, and a rename breaks the
#: test that pins it rather than quietly widening the answer.
THE_FOREGROUND_GUARDS: tuple[str, ...] = (
    "is_background",
    "_origin_is_user_facing",
    "clean_user_surface_contract",
)

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
    #: Files where every site for this channel sits under a guard that only a
    #: foreground user-facing turn passes. A background caller is in the file
    #: and never on the line.
    foreground_only_in: tuple[str, ...] = ()

    #: Harnesses whose caller is a background, non-user-facing request. A site
    #: behind a foreground guard is not on their path however completely the
    #: file is.
    BACKGROUND_HARNESSES = ("a background call through the gate",)

    def reachable_by(self, harness: str) -> bool:
        allowed = WHAT_A_HARNESS_REACHES.get(harness, ())
        if not allowed:
            return False
        where = self.applied_in
        if harness in self.BACKGROUND_HARNESSES:
            where = tuple(one for one in where if one not in self.foreground_only_in)
        return any(one.startswith(allowed) for one in where)

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "applied_in": list(self.applied_in),
            "foreground_only_in": list(self.foreground_only_in),
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
    # Per channel and file: whether every site there was under a foreground
    # guard. One unguarded site is enough to make the file reachable.
    guarded: dict[tuple[str, str], bool] = {}

    def note(key: str, where: str, under_a_guard: bool) -> None:
        sites.setdefault(key, set()).add(where)
        seen = guarded.get((key, where))
        guarded[(key, where)] = under_a_guard if seen is None else (seen and under_a_guard)

    for path in (root / "core").rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except (OSError, SyntaxError, ValueError):
            continue
        where = str(path.relative_to(root))
        for node, under_a_guard in _calls_with_their_guards(tree):
            if not node.args:
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
                note(key, where, under_a_guard)
    return tuple(
        AChannelSite(
            channel=name,
            applied_in=tuple(sorted(where)),
            foreground_only_in=tuple(
                sorted(one for one in where if guarded.get((name, one)))
            ),
        )
        for name, where in sorted(sites.items())
    )


def _calls_with_their_guards(tree: ast.AST) -> list[tuple[ast.Call, bool]]:
    """Every call, and whether a foreground-only test encloses it.

    Being in the gate's file is not being on the gate's background path. The
    circumplex is applied inside ``if not is_background and
    self._origin_is_user_facing(origin)``, and the campaign that asks the gate
    for a background generation is in the file and never on the line — which
    is why nine channels could rotate for three weeks and move none.

    Only lexical enclosure, only the tests named in
    :data:`THE_FOREGROUND_GUARDS`. A guard reached through a variable assigned
    twenty lines earlier is not seen, and reporting a site as reachable is the
    safe direction for a count that decides whether to spend model time.
    """
    found: list[tuple[ast.Call, bool]] = []

    def guardy(test: ast.AST) -> bool:
        source = ast.dump(test)
        return any(guard in source for guard in THE_FOREGROUND_GUARDS)

    def walk(node: ast.AST, under: bool) -> None:
        if isinstance(node, ast.Call):
            found.append((node, under))
        if isinstance(node, ast.If):
            inside = under or guardy(node.test)
            for child in ast.iter_child_nodes(node.test):
                walk(child, under)
            for statement in node.body:
                walk(statement, inside)
            for statement in node.orelse:
                walk(statement, under)
            return
        for child in ast.iter_child_nodes(node):
            walk(child, under)

    walk(tree, False)
    return found


def what_a_direct_call_can_bite(repo: str = ".") -> tuple[str, ...]:
    """Channels a harness that calls the model itself can actually move."""
    return tuple(
        one.channel
        for one in where_each_channel_acts(repo)
        if one.reachable_by("a direct model call")
    )


def what_the_probe_turn_can_bite(repo: str = ".") -> tuple[str, ...]:
    """Channels the campaign's measurement turn can actually move.

    The turn runs down the desktop quick-reply lane with a foreground origin,
    so both guards that shut the background caller out are open to it.
    """
    return tuple(
        one.channel
        for one in where_each_channel_acts(repo)
        if one.reachable_by("a turn through the probe lane")
    )


def what_a_background_gate_call_can_bite(repo: str = ".") -> tuple[str, ...]:
    """Channels the hourly influence campaign's own generator can move.

    The campaign asks the inference gate with a background, non-user-facing
    origin. A channel applied only inside the gate's foreground branch is not
    one of these, however completely the gate is on its path.
    """
    return tuple(
        one.channel
        for one in where_each_channel_acts(repo)
        if one.reachable_by("a background call through the gate")
    )


def how_the_lesions_are_reachable(repo: str = ".") -> dict[str, Any]:
    """For the health report: declared, applied, reachable by what, consumable how.

    Where a channel is applied is half of whether an arm can move it. The
    other half is what the channel hands over: a harness standing in the right
    place still cannot measure a value it has no way to consume, and the
    matched protocol reported one measurable faculty of ten for exactly that
    reason. Both halves are answered here so a protocol reads one block rather
    than inferring the second from the first.
    """
    from core.verify.what_a_channel_hands_over import (
        what_a_direct_harness_can_move,
        what_each_channel_hands_over,
    )

    every = where_each_channel_acts(repo)
    inert = [one.channel for one in every if not one.applied_in]
    handed = {one.channel: one for one in what_each_channel_hands_over()}
    movable = set(what_a_direct_harness_can_move())

    def _with_handover(one: AChannelSite) -> dict[str, Any]:
        said = one.to_dict()
        handover = handed.get(one.channel)
        if handover is not None:
            said["hands_over"] = list(handover.kinds)
            said["a_plain_call_can_move_it"] = handover.a_plain_call_can_move_it
            if handover.why_not:
                said["why_a_plain_call_cannot"] = handover.why_not
        return said

    return {
        "schema": "aura.lesions.reachability.v2",
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
        # What a harness that sets sampler parameters and writes a prompt can
        # vary, whatever else it boots. The number an ablation protocol is
        # entitled to claim before it runs.
        "a_plain_call_can_move": sorted(movable),
        "each": [_with_handover(one) for one in every],
    }
