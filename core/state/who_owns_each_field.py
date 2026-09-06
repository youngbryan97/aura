"""Who owns each durable field, and how it was worked out.

Generative Agents keeps its persona state small enough that one object owns
each fact. Aura's is 98 leaf fields across eight organs, which is right for
what it does and leaves a question the smaller system never has to ask: for
any one field, which thing decides its value?

Three answers count as owned, in this order:

1. A phase declares it in ``cognitive_contract``. The compiled plan already
   says which phases write which paths, so this needs no new declaration and
   cannot drift from what actually runs.
2. Exactly one module in ``core`` assigns it. That module is the owner in
   fact, whether or not anyone wrote it down.
3. It is in ``THE_DECLARED_OWNERS`` below, because several modules write it
   and one of them is the authority.

Anything else is unclassified: a field several things write with no stated
authority, or one nothing appears to write at all — which usually means it is
mutated in place, and a field mutated in place through a container reference
has no owner in any useful sense.

The measurement is deliberately limited to nested fields — ``cognition.mode``,
not ``version``. A bare attribute assignment to ``version`` or ``updated_at``
matches a dozen unrelated dataclasses, and a count built on that would be
confidently wrong. The eighteen top-level scalars are listed as out of reach
rather than guessed at.
"""
from __future__ import annotations

import ast
import dataclasses
import functools
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger("Aura.WhoOwnsEachField")

__all__ = [
    "THE_DECLARED_OWNERS",
    "every_field",
    "how_ownership_stands",
    "the_owner_of",
    "what_it_stood_at_last_time",
    "what_nobody_owns",
]

#: Fields several modules write, and which one is the authority. Each says
#: what the others are doing, because "one owner" is only true if the rest are
#: something other than writers.
THE_DECLARED_OWNERS: dict[str, dict[str, str]] = {
    "cognition.current_objective": {
        "owner": "core/phases/cognitive_routing.py",
        "others": "the kernel and mind_tick set it at the start of a turn and "
        "clear it at the end; executive_authority and executive_closure "
        "propose, response_policy reads it back",
    },
    "cognition.current_origin": {
        "owner": "core/phases/cognitive_routing.py",
        "others": "written beside current_objective by the same writers; it is "
        "one fact in two fields and they are never set apart",
    },
    "cognition.current_mode": {
        "owner": "core/phases/cognitive_routing.py",
        "others": "advisory_passes and phi_consciousness raise it for a turn; "
        "will_engine lowers it under refusal",
    },
    "cognition.attention_focus": {
        "owner": "core/consciousness/executive_closure.py",
        "others": "the assembler and advisory passes clear it when the thing "
        "attended to is gone",
    },
    "cognition.last_response": {
        "owner": "core/phases/response_generation.py",
        "others": "repair_phase replaces it with the repaired reply, "
        "response_policy blanks it on refusal",
    },
    "cognition.active_goals": {
        "owner": "core/consciousness/executive_closure.py",
        "others": "aura_state prunes them during trimming",
    },
    "identity.name": {
        "owner": "core/self/canonical_self.py",
        "others": "identity_reflection reads it back onto state after a "
        "reflection pass",
    },
    "identity.current_narrative": {
        "owner": "core/self/canonical_self.py",
        "others": "aura_state carries it across a derive",
    },
    "identity.stability": {
        "owner": "core/self/canonical_self.py",
        "others": "memory_consolidation lowers it when consolidation finds "
        "contradiction",
    },
    "affect.dominant_emotion": {
        "owner": "core/self/canonical_self.py",
        "others": "stream_of_being names it for the moment being narrated",
    },
}

#: The organs whose fields this can attribute. A top-level scalar on AuraState
#: cannot be: `x.version = 1` appears in a dozen unrelated dataclasses and
#: nothing in the assignment says which one it is.
_NESTED_ONLY = True


def every_field() -> list[str]:
    """Every leaf field of AuraState, as a dotted path."""
    from core.state.aura_state import AuraState

    def leaves(obj: Any, prefix: str = "") -> list[str]:
        found: list[str] = []
        for one in dataclasses.fields(obj):
            value = getattr(obj, one.name, None)
            if dataclasses.is_dataclass(value) and not isinstance(value, type):
                found.extend(leaves(value, f"{prefix}{one.name}."))
            else:
                found.append(f"{prefix}{one.name}")
        return found

    return leaves(AuraState())


@functools.lru_cache(maxsize=1)
def _declared_by_a_phase() -> frozenset[str]:
    from core.runtime.the_shape_of_one_turn import THE_MODES, compile_the_cognition

    written: set[str] = set()
    for mode in THE_MODES:
        for phase in compile_the_cognition(mode).phases:
            written.update(phase.writes)
    return frozenset(written)


@functools.lru_cache(maxsize=4)
def _who_assigns(root: Path) -> dict[str, frozenset[str]]:
    """Modules that assign each nested field, by exact two-segment match.

    Cached: this walks every file under ``core`` and parses it, which takes
    about twenty seconds. Called once per question it was answering the same
    question fourteen times over.
    """
    nested = {one for one in every_field() if one.count(".") == 1}
    found: dict[str, set[str]] = defaultdict(set)
    for path in sorted((root / "core").rglob("*.py")):
        if "__pycache__" in str(path):
            continue
        try:
            tree = ast.parse(path.read_text("utf-8", errors="ignore"))
        except (SyntaxError, OSError, ValueError):
            continue
        rel = str(path.relative_to(root))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AugAssign)):
                continue
            targets = (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
            for target in targets:
                if not isinstance(target, ast.Attribute):
                    continue
                parts: list[str] = []
                cursor: Any = target
                while isinstance(cursor, ast.Attribute):
                    parts.append(cursor.attr)
                    cursor = cursor.value
                parts.reverse()
                if len(parts) < 2:
                    continue
                dotted = ".".join(parts[-2:])
                if dotted in nested:
                    found[dotted].add(rel)
    return {path: frozenset(who) for path, who in found.items()}


@functools.lru_cache(maxsize=4)
def _stands(here: Path) -> dict[str, Any]:
    """Cached because the source does not change while the process runs.

    Without this the health report took eighteen seconds on its first call and
    two to four on every one after — a report that expensive stops being read,
    and the runtime calls it on a route.
    """
    return _work_out_how_ownership_stands(here)


def how_ownership_stands(root: Path | None = None) -> dict[str, Any]:
    """The whole picture, in the order the three answers are tried."""
    here = root or Path(__file__).resolve().parents[2]
    return dict(_stands(here))


def _work_out_how_ownership_stands(here: Path) -> dict[str, Any]:
    fields = every_field()
    nested = [one for one in fields if one.count(".") == 1]
    by_a_phase = _declared_by_a_phase()
    assigns = _who_assigns(here)

    phase_owned = [one for one in nested if one in by_a_phase]
    rest = [one for one in nested if one not in by_a_phase]
    single = [one for one in rest if len(assigns.get(one, ())) == 1]
    declared = [
        one
        for one in rest
        if one not in single and one in THE_DECLARED_OWNERS
    ]
    contested = [
        one
        for one in rest
        if one not in single
        and one not in THE_DECLARED_OWNERS
        and len(assigns.get(one, ())) > 1
    ]
    silent = [one for one in rest if not assigns.get(one)]

    return {
        "fields": len(fields),
        "nested": len(nested),
        "out_of_reach": len(fields) - len(nested),
        "owned_by_a_phase": sorted(phase_owned),
        "owned_by_one_module": sorted(single),
        "owned_by_declaration": sorted(declared),
        "written_by_several_with_no_authority": sorted(contested),
        "written_by_nothing_that_assigns": sorted(silent),
        "unclassified": len(contested) + len(silent),
    }


def the_owner_of(path: str, root: Path | None = None) -> dict[str, str] | None:
    """Who owns one field, and how that was worked out. None if nobody does."""
    here = root or Path(__file__).resolve().parents[2]
    if path in _declared_by_a_phase():
        from core.runtime.the_shape_of_one_turn import compile_the_cognition

        writers = [
            phase.phase
            for phase in compile_the_cognition("foreground").phases
            if path in phase.writes
        ]
        return {
            "owner": writers[-1] if writers else "a phase",
            "how": "declared in the phase contract",
        }
    assigns = _who_assigns(here).get(path, set())
    if len(assigns) == 1:
        return {"owner": next(iter(assigns)), "how": "the only module that assigns it"}
    declared = THE_DECLARED_OWNERS.get(path)
    if declared:
        return {
            "owner": declared["owner"],
            "how": "declared, because several write it",
            "others": declared["others"],
        }
    return None


def what_it_stood_at_last_time() -> dict[str, Any]:
    """The committed measurement, read from the baseline file.

    Cheap on purpose. Working it out means parsing every file under ``core``,
    which takes eleven seconds, and the health report is served on a route —
    a report that expensive stops being read. The baseline is the number this
    commit stands behind; ``how_ownership_stands`` is what the gate runs.
    """
    where = Path(__file__).resolve().parents[2] / "config" / "field_ownership_baseline.json"
    try:
        held = json.loads(where.read_text("utf-8"))
    except (OSError, ValueError) as exc:
        logger.debug("no field-ownership baseline: %s", exc)
        return {"unclassified": None, "note": "no baseline"}
    return {
        "unclassified": held.get("count"),
        "owned": held.get("owned"),
        "out_of_reach": held.get("out_of_reach"),
        "worked_out_this_process": _stands.cache_info().currsize > 0,
    }


def what_nobody_owns(root: Path | None = None) -> list[str]:
    """The gate. A field with no authority, and the baseline only shrinks."""
    stands = how_ownership_stands(root)
    return sorted(
        stands["written_by_several_with_no_authority"]
        + stands["written_by_nothing_that_assigns"]
    )
