"""What each lesion channel's value becomes, and who could consume it.

``which_lesions_a_direct_call_can_bite`` answers where a channel is applied.
That is half the question. A channel applied inside ``inference_gate`` is on
the path of any generation that goes through the gate, and still cannot be
measured by a harness that has no way to consume what the channel hands over.
The matched protocol reported one measurable faculty out of ten for exactly
that reason, and the reading was correct without being useful: it said the
harness was thin without saying what would thicken it.

So this asks the other half. A channel hands over a value, that value lands
somewhere, and the name of where it lands says what kind of thing it is:

* ``sampler``  — temperature, top_p, a token budget, a logit bias. Anything a
  plain call to the model already takes as an argument.
* ``prompt``   — text that goes into the context window.
* ``loop``     — how many recurrent passes to run, how hard to steer them.
  Needs a decoder that loops; a single forward pass cannot consume it.
* ``downstream`` — a number some other faculty reads. No generation sees it.

A harness that can set sampler parameters and write a prompt can move every
``sampler`` and ``prompt`` channel and none of the others, whatever else it
boots. That is the number worth reporting, and it is computed here rather
than declared: the destinations come out of the source at each site, and a
destination whose name matches nothing is reported as ``unclassified`` and
counted, so a new consumer arrives loudly instead of being absorbed.

Three ways a channel binds and all three are read. ``apply_channel`` passes a
value through and the destination is wherever that value is stored.
``is_lesioned`` guards a block and the destinations are what the block writes;
when the guard's result is kept in a local, the ifs that test that local
count too. ``@lesionable`` binds the channel to a class's own ``lesion()``
and leaves nothing in the source to follow, so those are reported honestly as
having no destination found rather than being given one by hand.
"""
from __future__ import annotations

import ast
import functools
import logging
import pathlib
from dataclasses import dataclass

logger = logging.getLogger("Aura.WhatAChannelHandsOver")

__all__ = [
    "AHandover",
    "WHAT_A_NAME_MEANS",
    "WHAT_A_PLAIN_CALL_CONSUMES",
    "what_each_channel_hands_over",
    "what_a_direct_harness_can_move",
    "how_the_handovers_read",
]

#: The kinds, and the substrings in a destination's name that identify each.
#: Checked in this order: ``loop`` first, because a recurrent-pass count is a
#: number a sampler-shaped name would otherwise swallow.
WHAT_A_NAME_MEANS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("loop", ("recurrent", "steering", "_loops", "passes")),
    (
        "sampler",
        (
            "temperature",
            "temp",
            "top_p",
            "top_k",
            "max_tokens",
            "num_predict",
            "sampling_bias",
            "repetition",
            "token_budget",
        ),
    ),
    ("prompt", ("prompt", "context", "grounding", "block", "system_message")),
)

#: What a harness that calls the model directly is able to vary. A channel
#: that hands over one of these can be measured without booting anything that
#: loops or anything downstream of the reply.
WHAT_A_PLAIN_CALL_CONSUMES: frozenset[str] = frozenset({"sampler", "prompt"})

_WHERE_TO_LOOK: tuple[str, ...] = (
    "core/brain/cognitive_engine.py",
    "core/brain/inference_gate.py",
    "core/consciousness/qualia_synthesizer.py",
    "core/affect/affective_circumplex.py",
    "core/being/affective_valence.py",
)


@dataclass(frozen=True, slots=True)
class AHandover:
    """One channel, what its value becomes, and who could consume it."""

    channel: str
    bound_by: tuple[str, ...]
    destinations: tuple[str, ...]
    kinds: tuple[str, ...]

    @property
    def unclassified(self) -> tuple[str, ...]:
        """Destinations found in the source whose name matched no kind."""
        return tuple(d for d in self.destinations if _kind_of(d) is None)

    @property
    def a_plain_call_can_move_it(self) -> bool:
        return bool(WHAT_A_PLAIN_CALL_CONSUMES.intersection(self.kinds))

    @property
    def why_not(self) -> str:
        if self.a_plain_call_can_move_it:
            return ""
        if not self.destinations:
            return (
                f"bound by {'/'.join(self.bound_by) or 'nothing found'} and the "
                "source carries no destination to follow"
            )
        return (
            "everything it hands over is "
            + "/".join(self.kinds or ("unclassified",))
            + ": no sampler parameter and no prompt text"
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "channel": self.channel,
            "bound_by": list(self.bound_by),
            "destinations": list(self.destinations),
            "kinds": list(self.kinds),
            "unclassified": list(self.unclassified),
            "a_plain_call_can_move_it": self.a_plain_call_can_move_it,
            "why_not": self.why_not,
        }


def _kind_of(destination: str) -> str | None:
    low = destination.lower()
    for kind, marks in WHAT_A_NAME_MEANS:
        if any(mark in low for mark in marks):
            return kind
    return None


def _name_of(node: ast.AST) -> str | None:
    """A readable name for an assignment target or a dict key."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _name_of(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Subscript):
        base = _name_of(node.value) or ""
        key = node.slice
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            return f"{base}[{key.value}]"
        return base or None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _channel_named(node: ast.AST, constants: dict[str, str]) -> str | None:
    """Resolve ``influence_channels.X`` (or a bare ``X``) to the channel id."""
    if isinstance(node, ast.Attribute) and node.attr in constants:
        return constants[node.attr]
    if isinstance(node, ast.Name) and node.id in constants:
        return constants[node.id]
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value if node.value in constants.values() else None
    return None


@functools.lru_cache(maxsize=1)
def _channel_constants() -> dict[str, str]:
    from core.verify import influence_channels

    return {
        name: value
        for name, value in vars(influence_channels).items()
        if name.isupper() and isinstance(value, str) and "." in value
    }


def _targets_written_in(node: ast.AST) -> list[str]:
    found: list[str] = []
    for inner in ast.walk(node):
        targets: list[ast.AST] = []
        if isinstance(inner, ast.Assign):
            targets = list(inner.targets)
        elif isinstance(inner, (ast.AnnAssign, ast.AugAssign)):
            targets = [inner.target]
        for target in targets:
            name = _name_of(target)
            if name:
                found.append(name)
    return found


def _walk_with_parents(tree: ast.AST) -> dict[int, ast.AST]:
    parents: dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node
    return parents


def _destination_of_a_value(node: ast.AST, parents: dict[int, ast.AST]) -> str | None:
    """Where the value produced at ``node`` is stored.

    Walks out through the calls that wrap it — ``int(...)``, ``max(...)`` —
    because a budget clamped on its way to a keyword is still that keyword.
    """
    seen = 0
    current: ast.AST | None = node
    while current is not None and seen < 12:
        seen += 1
        parent = parents.get(id(current))
        if parent is None:
            return None
        if isinstance(parent, ast.Dict):
            for key, value in zip(parent.keys, parent.values, strict=False):
                if value is current and key is not None:
                    return _name_of(key)
            return None
        if isinstance(parent, ast.Assign):
            for target in parent.targets:
                name = _name_of(target)
                if name:
                    return name
            return None
        if isinstance(parent, (ast.AnnAssign, ast.AugAssign)):
            return _name_of(parent.target)
        if isinstance(parent, ast.keyword) and parent.arg:
            return parent.arg
        current = parent
    return None


def _sites_in(path: pathlib.Path, constants: dict[str, str]) -> dict[str, tuple[set[str], set[str]]]:
    """Per channel in this file: (how it binds, where its value lands)."""
    out: dict[str, tuple[set[str], set[str]]] = {}

    def note(channel: str, bound: str, destinations: list[str]) -> None:
        binds, dests = out.setdefault(channel, (set(), set()))
        binds.add(bound)
        dests.update(d for d in destinations if d)

    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except (OSError, SyntaxError, ValueError) as exc:
        logger.debug("could not read %s: %s", path, exc)
        return out
    parents = _walk_with_parents(tree)

    # A guard whose verdict is kept in a local: every if in the file that
    # tests that local is part of what the channel decides.
    ifs_by_test: dict[str, list[ast.If]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            for inner in ast.walk(node.test):
                if isinstance(inner, ast.Name):
                    ifs_by_test.setdefault(inner.id, []).append(node)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        called = node.func.attr if isinstance(node.func, ast.Attribute) else (
            node.func.id if isinstance(node.func, ast.Name) else ""
        )
        if called == "apply_channel" and node.args:
            channel = _channel_named(node.args[0], constants)
            if channel:
                where = _destination_of_a_value(node, parents)
                note(channel, "apply_channel", [where] if where else [])
        elif called == "is_lesioned" and node.args:
            channel = _channel_named(node.args[0], constants)
            if not channel:
                continue
            destinations: list[str] = []
            parent = parents.get(id(node))
            while parent is not None and not isinstance(
                parent, (ast.If, ast.Assign, ast.Module)
            ):
                parent = parents.get(id(parent))
            if isinstance(parent, ast.If):
                destinations.extend(_targets_written_in(parent))
            elif isinstance(parent, ast.Assign):
                for target in parent.targets:
                    local = _name_of(target)
                    if not local:
                        continue
                    for guarded in ifs_by_test.get(local, ()):
                        destinations.extend(_targets_written_in(guarded))
            note(channel, "is_lesioned", destinations)
        elif called == "lesionable" and node.args:
            channel = _channel_named(node.args[0], constants)
            if channel:
                note(channel, "lesionable", [])
        elif called == "register_flag_lesion" and node.args:
            channel = _channel_named(node.args[0], constants)
            if channel:
                note(channel, "register_flag_lesion", [])

    return out


@functools.lru_cache(maxsize=1)
def what_each_channel_hands_over() -> tuple[AHandover, ...]:
    """Every declared channel, with what its value becomes."""
    from core.verify import influence_channels

    constants = _channel_constants()
    root = pathlib.Path(__file__).resolve().parents[2]
    binds: dict[str, set[str]] = {}
    dests: dict[str, set[str]] = {}
    for relative in _WHERE_TO_LOOK:
        for channel, (bound, destinations) in _sites_in(root / relative, constants).items():
            binds.setdefault(channel, set()).update(bound)
            dests.setdefault(channel, set()).update(destinations)

    out: list[AHandover] = []
    for channel in sorted(influence_channels.ALL_CHANNELS):
        destinations = tuple(sorted(dests.get(channel, set())))
        kinds = tuple(
            sorted({kind for kind in map(_kind_of, destinations) if kind is not None})
        )
        out.append(
            AHandover(
                channel=channel,
                bound_by=tuple(sorted(binds.get(channel, set()))),
                destinations=destinations,
                kinds=kinds,
            )
        )
    return tuple(out)


def what_a_direct_harness_can_move() -> tuple[str, ...]:
    """The channels a harness that samples and prompts is entitled to claim."""
    return tuple(
        h.channel for h in what_each_channel_hands_over() if h.a_plain_call_can_move_it
    )


def how_the_handovers_read() -> dict[str, object]:
    """One block for the inspector and the health report."""
    handovers = what_each_channel_hands_over()
    movable = [h for h in handovers if h.a_plain_call_can_move_it]
    unclassified = sorted(
        {d for h in handovers for d in h.unclassified}
    )
    by_kind: dict[str, int] = {}
    for handover in handovers:
        for kind in handover.kinds or ("nothing found",):
            by_kind[kind] = by_kind.get(kind, 0) + 1
    return {
        "declared": len(handovers),
        "a_plain_call_can_move": len(movable),
        "by_kind": dict(sorted(by_kind.items())),
        "unclassified_destinations": unclassified,
        "channels": [h.as_dict() for h in handovers],
    }
