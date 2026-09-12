"""How many named faculties have a measured downstream effect. Today: none.

An external review wrote the distinction that governs this whole file:

    a channel wired to a consumer is not a measured downstream effect.

Aura declares 69 services and has a causal-influence framework that requires
treatment against null, a lesion registry that can run a turn without a
faculty, and an influence ledger that returns UNMEASURED, INERT or
INFLUENTIAL. What it does not have is coverage: seven channels are lesionable,
none of them has been measured, and two modules in the tree register a lesion
at all.

That number is what the framework exists to produce, and publishing it is the
difference between having a standard and meeting it.

Three counts, and they move in known directions:

* ``lesionable`` — faculties a turn can be run without. Goes up.
* ``measured`` — those with enough paired trials to resolve an effect from
  noise. Goes up.
* ``unmeasured`` — lesionable and never put through a treatment and a null.
  Goes down.

A faculty that is not lesionable is not counted as unmeasured, because it
cannot be measured at all. It is counted as ``not_lesionable``, which is the
larger number and the one that has to fall first.
"""
from __future__ import annotations

import functools
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("Aura.WhatHasAMeasuredEffect")

__all__ = [
    "how_much_is_measured",
    "what_it_stood_at_last_time",
    "the_declared_lesions",
    "the_baseline",
    "what_is_still_unmeasured",
    "what_the_substrate_trials_found",
]

BASELINE = Path(__file__).resolve().parents[2] / "config" / "measured_effect_baseline.json"

#: Verdicts from paired trials run against a named substrate by
#: ``tools/run_influence_trials_on_the_substrate.py``. A separate count from
#: the live one and it must stay separate: a verdict at Qwen2.5-1.5B is not a
#: verdict at the 27B, and merging the two numbers would let the small model
#: discharge the large model's obligation.
SUBSTRATE_TRIALS = (
    Path(__file__).resolve().parents[2] / "artifacts" / "influence" / "substrate_trials.json"
)


#: Every spelling of "make this channel lesionable". A registration through
#: one of these is what makes a faculty falsifiable at all.
#:
#: ``lesionable`` is the decorator form, and leaving it out was the same
#: mistake this file warns about elsewhere: the affective valence engine binds
#: its channel to the class's own ``lesion()`` method, which is a registration
#: by any reading, and the count said the channel did not exist.
_THE_WAYS_TO_REGISTER: frozenset[str] = frozenset(
    {"register_lesion", "register_flag_lesion", "register_value_lesion", "lesionable"}
)


def _a_channel_name(node: Any, path: Path) -> str:
    """The channel a registration names, literal or through a constant.

    ``influence_channels.LIVE_MIND_STEERING_ALPHA`` is the usual form. The
    attribute's own name is what identifies it here; resolving it to the
    string would mean importing the module, and importing to count is how the
    count became import-order dependent in the first place.
    """
    import ast

    if isinstance(node, ast.Constant):
        return str(node.value)
    if isinstance(node, ast.Attribute):
        return str(node.attr)
    # A bare name is a variable — a wrapper passing its own argument through.
    # Counting it added a channel called "channel" to the list.
    return ""


def _channels_a_loop_binds(tree: Any) -> dict[str, set[str]]:
    """Loop variables that take a channel name, and which names they take.

    Three of these are registered in a ``for`` over a literal tuple of
    ``(channel, source)`` pairs. The argument at the call is then a bare
    ``ast.Name``, which this file deliberately refuses to count — so the
    spiking, imagination and bicameral sampling biases were registered in the
    source, registered at runtime, and absent from the declared count. The
    ratchet undercounted the apparatus by three.

    Only literal iterables. A loop over something computed is a channel set
    this cannot know statically, and guessing at one is how a static count
    starts disagreeing with the runtime it is meant to describe.
    """
    import ast

    bound: dict[str, set[str]] = {}

    def elements(node: Any) -> list[Any]:
        return list(node.elts) if isinstance(node, (ast.Tuple, ast.List)) else []

    for node in ast.walk(tree):
        if not isinstance(node, ast.For):
            continue
        rows = elements(node.iter)
        if not rows:
            continue
        targets = (
            [node.target] if isinstance(node.target, ast.Name) else elements(node.target)
        )
        for position, target in enumerate(targets):
            if not isinstance(target, ast.Name):
                continue
            for row in rows:
                cells = elements(row) if len(targets) > 1 else [row]
                if position >= len(cells):
                    continue
                name = _a_channel_name(cells[position], Path("."))
                if name:
                    bound.setdefault(target.id, set()).add(name)
    return bound


def _every_name_for(node: Any, path: Path, bound: dict[str, set[str]]) -> set[str]:
    """The channel names this argument can carry, literal or loop-bound."""
    import ast

    if isinstance(node, ast.Name) and node.id in bound:
        return set(bound[node.id])
    return {_a_channel_name(node, path)}


def the_declared_lesions(root: Path | None = None) -> list[str]:
    """See ``_declared_lesions``; this is the cached front door."""
    return list(_declared_lesions(root or Path(__file__).resolve().parents[2]))


@functools.lru_cache(maxsize=4)
def _declared_lesions(here: Path) -> tuple[str, ...]:
    """Every channel the tree registers a lesion for, read from the source.

    Static on purpose. Counting the live registry counts whatever this process
    happened to import — the first version of this said 7 in one process and 1
    in another, which makes a ratchet on it worse than no ratchet. What is
    declared in the source is the same number every time.

    Cached: it parses every file under ``core``, and the health report asks.
    """
    import ast

    found: set[str] = set()
    for path in sorted((here / "core").rglob("*.py")):
        if "__pycache__" in str(path) or path.name == "lesion_registry.py":
            continue
        try:
            tree = ast.parse(path.read_text("utf-8", errors="ignore"))
        except (SyntaxError, OSError, ValueError):
            continue
        bound = _channels_a_loop_binds(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name not in _THE_WAYS_TO_REGISTER:
                continue
            # The channel is a name from `influence_channels` far more often
            # than a literal, so the attribute counts. Counting only literals
            # said zero while seven were registered.
            for keyword in node.keywords:
                if keyword.arg == "channel":
                    found.update(_every_name_for(keyword.value, path, bound))
            for argument in node.args[:1]:
                found.update(_every_name_for(argument, path, bound))
    found.discard("")
    return tuple(sorted(found))


def how_much_is_measured() -> dict[str, Any]:
    """What the influence ledger can say about each lesionable faculty."""
    try:
        from core.service_names import ServiceNames
        from core.verify.causal_influence import Verdict, get_influence_ledger
        from core.verify.lesion_registry import get_lesion_registry
    except (ImportError, RuntimeError) as exc:
        return {"error": repr(exc)}

    declared_services = sorted(
        value
        for key, value in vars(ServiceNames).items()
        if not key.startswith("_") and isinstance(value, str)
    )
    registry = get_lesion_registry()
    ledger = get_influence_ledger()
    # The union: what the source declares, plus anything this process
    # registered dynamically. The declared set is what the ratchet counts.
    declared = the_declared_lesions()
    lesionable = sorted(set(declared) | set(registry.channels()))

    by_verdict: dict[str, list[str]] = {}
    for channel in lesionable:
        try:
            verdict = str(ledger.verdict(channel).verdict)
        except Exception:  # noqa: BLE001 — an unaskable channel is unmeasured
            verdict = str(Verdict.UNMEASURED)
        by_verdict.setdefault(verdict, []).append(channel)

    unmeasured = by_verdict.get(str(Verdict.UNMEASURED), [])
    influential = by_verdict.get(str(Verdict.INFLUENTIAL), [])
    inert = by_verdict.get(str(Verdict.INERT), [])
    on_the_substrate = what_the_substrate_trials_found()
    # The declared list carries constant names and the ledger carries channel
    # ids, so without this the two halves of the report cannot be joined:
    # AFFECT_CIRCUMPLEX_SAMPLING would read as unmeasured beside a verdict for
    # affect.circumplex_sampling and nobody could tell they were one channel.
    with_a_substrate_verdict = sorted(
        name
        for name in unmeasured
        if _the_channel_id(name) in on_the_substrate["channels"]
    )
    return {
        "declared_services": len(declared_services),
        "declared_lesions": len(declared),
        "lesionable": len(lesionable),
        "not_lesionable": max(0, len(declared_services) - len(lesionable)),
        "measured": len(influential) + len(inert),
        "unmeasured": len(unmeasured),
        "influential": sorted(influential),
        "inert": sorted(inert),
        "still_unmeasured": sorted(unmeasured),
        # Held apart from `measured` on purpose. These are verdicts, reached
        # by paired trials against a real model, and they are verdicts at a
        # 1.5B — which is a boundary, not a hedge.
        "measured_at_substrate": on_the_substrate["measured"],
        "substrate": on_the_substrate["substrate"],
        "substrate_verdicts": dict(on_the_substrate["channels"]),
        "measured_at_substrate_not_live": with_a_substrate_verdict,
        # The sentence the counts are for.
        "what_this_means": (
            "a channel wired to a consumer is not a measured downstream "
            "effect; only `measured` is evidence"
        ),
    }


@functools.lru_cache(maxsize=1)
def _by_constant_name() -> dict[str, str]:
    """Constant name to channel id, from the one module that owns both.

    Importing a leaf of constants is not the import-order problem the counting
    above avoids: nothing registers a lesion on the way in.
    """
    from core.verify import influence_channels

    return {
        name: value
        for name, value in vars(influence_channels).items()
        if name.isupper() and isinstance(value, str)
    }


def _the_channel_id(constant_or_id: str) -> str:
    return _by_constant_name().get(constant_or_id, constant_or_id)


def what_the_substrate_trials_found() -> dict[str, Any]:
    """Channels with a verdict from paired trials, and at which substrate.

    The live campaign measures nothing today and the reason is structural
    rather than statistical: its generator asks the gate for a background
    generation, and every ``apply_channel`` site sits behind a guard that a
    background call does not pass — the circumplex behind
    ``not is_background and origin_is_user_facing``, the live-mind and
    sampling-bias sites inside the clean-user-surface contract. So the hourly
    job was rotating through nine channels none of which its own generator
    could move.

    A direct call CAN move four of them, because the advisory frames, the
    circumplex and the production sampling fold all run without a turn. That
    is what these trials are, and the substrate is named in the result
    because it is a boundary: 1.5B, forty paired trials, the token budget
    free to move because half of each channel is the budget.
    """
    try:
        payload = json.loads(SUBSTRATE_TRIALS.read_text("utf-8"))
    except (OSError, ValueError) as exc:
        logger.debug("no substrate influence trials: %s", exc)
        return {"substrate": "", "measured": 0, "channels": {}}
    if not isinstance(payload, dict):
        return {"substrate": "", "measured": 0, "channels": {}}
    findings = payload.get("findings")
    channels: dict[str, str] = {}
    if isinstance(findings, dict):
        for channel, finding in findings.items():
            verdict = (finding or {}).get("verdict") if isinstance(finding, dict) else None
            name = str((verdict or {}).get("verdict") or "").lower()
            if name in {"influential", "inert"}:
                channels[str(channel)] = name
    return {
        "substrate": str(payload.get("substrate") or ""),
        "trials_per_channel": payload.get("trials_per_channel"),
        "measured": len(channels),
        "channels": channels,
        "unreachable_from_a_direct_call": list(
            payload.get("not_measurable_by_a_direct_call") or ()
        ),
        "what_this_is_not": str(payload.get("what_this_is_not") or ""),
    }


def what_is_still_unmeasured() -> list[str]:
    """Lesionable faculties with no treatment and no null. Goes down."""
    return list(how_much_is_measured().get("still_unmeasured") or ())


def what_it_stood_at_last_time() -> dict[str, Any]:
    """The committed counts, read from the baseline file.

    Cheap on purpose. Working it out parses every file under ``core``, which
    takes eight seconds, and the health report is served on a route — a report
    that expensive stops being read. ``how_much_is_measured`` is what the gate
    runs; this is what health shows.
    """
    held = the_baseline()
    if not held:
        return {"measured": None, "note": "no baseline"}
    return {
        "declared_lesions": held.get("declared_lesions"),
        "measured": held.get("measured"),
        # Separate, and named with its substrate, because a verdict at a 1.5B
        # is not a verdict at the cortex. Health showed only `measured`, which
        # is still zero, so four real verdicts were invisible where the rest of
        # the integrity block is read.
        "measured_at_substrate": held.get("measured_at_substrate"),
        "substrate": held.get("substrate"),
        "substrate_verdicts": dict(held.get("substrate_verdicts") or {}),
        "unmeasured": len(held.get("still_unmeasured") or ()),
        "still_unmeasured": list(held.get("still_unmeasured") or ())[:20],
        "worked_out_this_process": _declared_lesions.cache_info().currsize > 0,
        "what_this_means": (
            "a channel wired to a consumer is not a measured downstream "
            "effect; only `measured` is evidence"
        ),
    }


def the_baseline() -> dict[str, Any]:
    try:
        return json.loads(BASELINE.read_text("utf-8"))
    except (OSError, ValueError) as exc:
        logger.debug("no measured-effect baseline: %s", exc)
        return {}
