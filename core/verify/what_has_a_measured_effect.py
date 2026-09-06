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
]

BASELINE = Path(__file__).resolve().parents[2] / "config" / "measured_effect_baseline.json"


#: Every spelling of "make this channel lesionable". A registration through
#: one of these is what makes a faculty falsifiable at all.
_THE_WAYS_TO_REGISTER: frozenset[str] = frozenset(
    {"register_lesion", "register_flag_lesion", "register_value_lesion"}
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
                    found.add(_a_channel_name(keyword.value, path))
            for argument in node.args[:1]:
                found.add(_a_channel_name(argument, path))
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
        # The sentence the counts are for.
        "what_this_means": (
            "a channel wired to a consumer is not a measured downstream "
            "effect; only `measured` is evidence"
        ),
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
