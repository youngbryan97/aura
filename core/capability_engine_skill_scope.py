"""What a skill may reach, whether it was forged, and what is left when one is denied.

Lifted whole out of `capability_engine`, which imports them straight back: every
caller and every patch that names them there still finds them. What they
take from that module is imported at CALL time, for the same reason.
"""
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation


def _skill_reaches_beyond_its_scope(skill_class: Any, declared: str) -> str:
    """"" when the declaration covers the module, the refusal otherwise.

    Measured from the module's source, so it costs one AST parse per skill at
    registration and nothing afterwards. Mismatches that predate this check
    are grandfathered in config/skill_effect_scope_baseline.json, which only
    shrinks; a skill added after it must declare what it reaches for.
    """
    from .capability_engine import (
        _grandfathered_overreach,
    )

    try:
        from core.skills.effect_reach import measure_file, measure_source, violation

        module_file = inspect.getsourcefile(skill_class)
        if not module_file:
            return ""
        # A skill's helpers usually sit beside it at module level, so the
        # module is the right unit — but only when the module IS a skill
        # module. A class defined inside a test file or a fixture would
        # otherwise inherit that file's imports, which is how a test double
        # named `runtime_instance_skill` came to "reach" the file-write
        # gateway that the test itself imports.
        resolved = Path(module_file).resolve()
        in_skill_tree = any(
            part in {"skills"} for part in resolved.parts
        ) and "tests" not in resolved.parts
        if in_skill_tree:
            reach = measure_file(resolved)
        else:
            try:
                reach = measure_source(inspect.getsource(skill_class))
            except (OSError, TypeError):
                return ""
        problem: str = violation(declared, reach)
        if not problem:
            return ""
        key = f"{getattr(skill_class, 'name', '')}:{declared}"
        if any(entry.startswith(key + "->") for entry in _grandfathered_overreach()):
            return ""
        return problem
    except (ImportError, OSError, TypeError, ValueError) as exc:
        # A check that cannot run has not cleared anything, but refusing every
        # skill because the analyser broke would take the whole catalog down.
        record_degradation(
            "capability_engine.effect_scope",
            exc,
            severity="warning",
            action="registered the skill without comparing its scope with its reach",
        )
        return ""


def _name_what_is_still_available(
    denial: dict[str, Any], refused: Any, context: dict[str, Any] | None
) -> dict[str, Any]:
    """Add the other tools this turn was offered to a refusal.

    LIVE, 2026-08-27: file_operation was leased read_only for the turn, and the
    model asked it to WRITE a script so it could exercise a library — twice.
    Both calls came back "denied_by_default: tool_execution requires validated
    scoped authority", which says what went wrong and nothing about what would
    work, while code_repl sat offered on the same turn, able to import that
    library and run it.

    Naming what remains is not a hint about the task. It is the part of a
    refusal the caller can act on, and the complement of not offering a tool
    that will be refused.
    """
    offered = (context or {}).get("required_skills") or (context or {}).get("offered_skills")
    if not isinstance(offered, (list, tuple, set)):
        return denial
    name = str(refused or "")
    instead = sorted({str(item) for item in offered if str(item) and str(item) != name})[:4]
    if not instead:
        return denial
    told = dict(denial)
    told["available_instead"] = instead
    told["error"] = f"{told.get('error', 'refused')}. Still available on this turn: " + ", ".join(
        instead
    ) + "."
    return told


def _is_forged_skill(meta: Any) -> bool:
    """Whether this skill is code Aura wrote for herself.

    The predicate used to be ``"skills/" in meta.module_path``. Module paths are
    dotted — ``skills.word_count`` — so the substring never matched and the
    answer was always no. Everything downstream of it was therefore dead: the
    Sandbox 2.0 branch never ran, and model-authored code executed in-process
    like any hand-written skill, under a log line announcing that it was
    confined.

    ``source_kind`` is the catalog's own answer to the same question, set by
    :func:`core.skills.discovery.default_skill_roots` when it walks the writable
    ``skills/`` tree rather than ``core/skills``. It is a fact about where the
    file came from instead of a guess from how a string is spelled.
    """
    if str(getattr(meta, "source_kind", "") or "").strip().lower() == "project":
        return True
    # A skill registered at runtime carries no catalog provenance. Fall back to
    # the module's real file, which is a path and can be compared as one.
    module_path = str(getattr(meta, "source_path", "") or "")
    if not module_path:
        return False
    return Path(module_path).as_posix().startswith("skills/")
