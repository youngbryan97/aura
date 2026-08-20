"""What a CALL does, not what its skill is capable of.

A skill's ``effect_scope`` is the worst thing it can do. ``file_operation``
declares ``state_mutation`` because it can write, append, move and delete —
and so reading a file requires write authority, which is how a runtime with a
file reader ends up unable to look at anything.

    "Before she can debug an unfamiliar repository, analyse a paper, or check
    a spreadsheet, she has to be able to READ a file without being granted
    permission to destroy one."

Every one of those tasks begins by reading something. Scoping by skill rather
than by action put the cheapest, safest step in computing behind the most
dangerous grant in the system.

So a skill may declare what each of its actions actually does. The declared
skill scope stays the CEILING — an action can never claim less containment
than its skill was admitted for in a stricter direction — and anything the
skill does not describe keeps the skill's own scope, so silence is never read
as safety.

This is the mechanism only. Offering a skill because one of its actions is
harmless is safe only when the dispatch refuses the actions that are
not, so :func:`action_within_scope` is checked at execution, not merely
consulted when choosing what to offer.
"""

from __future__ import annotations

import importlib
from functools import lru_cache
from typing import Any

__all__ = [
    "EFFECT_SCOPE_RANK",
    "action_effect_scope",
    "action_field_and_default",
    "action_within_scope",
    "declared_action_name",
    "declared_action_scopes",
    "resolve_skill_target",
    "skill_class_named",
    "skill_has_action_within",
]


@lru_cache(maxsize=256)
def _import_class(module_path: str, class_name: str) -> Any:
    try:
        return getattr(importlib.import_module(module_path), class_name, None)
    except Exception:  # noqa: BLE001 - a skill that cannot import declares nothing
        return None


@lru_cache(maxsize=1)
def _declared_classes() -> dict[str, tuple[str, str]]:
    """Where each skill's class lives, by skill name.

    Built from the source catalogue, which parses declarations without
    importing or instantiating anything, so asking for one skill's scope does
    not start seventy-five others.
    """
    try:
        from core.skills.discovery import build_skill_catalog

        catalog = build_skill_catalog()
    except Exception:  # noqa: BLE001 - no catalogue means no declaration to read
        return {}
    found: dict[str, tuple[str, str]] = {}
    for declaration in getattr(catalog, "accepted", ()) or ():
        name = str(getattr(declaration, "name", "") or "").strip()
        module_path = str(getattr(declaration, "module_path", "") or "").strip()
        class_name = str(getattr(declaration, "class_name", "") or "").strip()
        if name and module_path and class_name:
            found.setdefault(name, (module_path, class_name))
    return found


def skill_class_named(name: Any) -> Any:
    """The class declaring a skill's actions, found by the skill's name.

    The registry is the fast path and is absent outside a running container.
    Scope resolution has to answer either way: a skill's declaration of what
    its own actions cost does not depend on whether the runtime is up.
    """
    wanted = str(name or "").strip()
    if not wanted:
        return None
    located = _declared_classes().get(wanted)
    if not located:
        return None
    return _import_class(*located)


def resolve_skill_target(meta: Any) -> Any:
    """The class that declares the actions, without instantiating the skill.

    Registry metadata carries `skill_class` and `instance` as None until the
    skill is first used, so reading the declaration off either of them finds
    nothing and every action falls back to the skill's worst-case scope — which
    is exactly the failure this module exists to remove. The module path is
    populated from registration, so the class is reachable by import.
    """
    for candidate in (getattr(meta, "skill_class", None), getattr(meta, "instance", None)):
        if candidate is not None and declared_action_scopes(candidate):
            return candidate
    module_path = str(getattr(meta, "module_path", "") or "").strip()
    class_name = str(getattr(meta, "class_name", "") or "").strip()
    if module_path and class_name:
        resolved = _import_class(module_path, class_name)
        if resolved is not None:
            return resolved
    by_name = skill_class_named(getattr(meta, "name", None))
    if by_name is not None:
        return by_name
    return meta

def _literal_choices(annotation: Any) -> set[str]:
    """The string values a Literal annotation allows, at any nesting."""
    from typing import Literal, Union, get_args, get_origin

    origin = get_origin(annotation)
    if origin is Literal:
        return {str(value).strip().lower() for value in get_args(annotation) if isinstance(value, str)}
    if origin is Union:
        found: set[str] = set()
        for argument in get_args(annotation):
            found |= _literal_choices(argument)
        return found
    return set()


def action_field_and_default(target: Any) -> tuple[str, str]:
    """Which input field names the action, and what it means when omitted.

    Found by matching the input model against the action table rather than by
    a list of field names: the field whose allowed values ARE the declared
    actions is the one that selects them, whatever it is called. file_operation
    calls it ``action`` and http_request calls it ``method``, and neither has
    to say so twice.

    The default matters as much as the name. A model that omits an optional
    field means the default, so a fetch with no method named is a GET — and
    reading the field as absent scoped that GET as the skill's worst action
    and refused it.
    """
    declared = set(declared_action_scopes(target))
    if not declared:
        return "", ""
    model = getattr(target, "input_model", None)
    fields = getattr(model, "model_fields", None) or {}
    for name, field in fields.items():
        choices = _literal_choices(getattr(field, "annotation", None))
        if choices and choices <= declared:
            default = getattr(field, "default", None)
            return str(name), str(default).strip().lower() if isinstance(default, str) else ""
    return "", ""


def declared_action_name(target: Any, args: Any) -> str:
    """Which of the skill's actions this call performs.

    Empty only when the call names no action the skill declares and the skill
    has no default, which is the case that must keep the skill's own scope.
    """
    field, default = action_field_and_default(target)
    arguments = args if isinstance(args, dict) else {}
    if field:
        value = arguments.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    declared = set(declared_action_scopes(target))
    for value in arguments.values():
        if isinstance(value, str) and value.strip().lower() in declared:
            return value.strip().lower()
    return default


#: Containment, least to most dangerous. A call is admissible when its scope
#: ranks no higher than what the turn is authorised for.
EFFECT_SCOPE_RANK: dict[str, int] = {
    "status": 0,
    "pure_compute": 1,
    "read_only": 2,
    "sandboxed_compute": 3,
    "read_write_artifacts": 4,
    "state_mutation": 5,
    "foreground_browser_dialogue": 5,
    "foreground_desktop_control": 6,
    "external_io": 7,
    "privileged_mutation": 8,
    "unknown": 9,
}


def _rank(scope: Any) -> int:
    return EFFECT_SCOPE_RANK.get(str(scope or "unknown").strip().lower(), 9)


def declared_action_scopes(target: Any) -> dict[str, str]:
    """What the skill says each of its actions does, or an empty map.

    Read off the skill itself so the answer lives beside the code that
    performs the action, rather than in a table somewhere else that drifts.
    """
    declared = getattr(target, "ACTION_EFFECT_SCOPES", None)
    if not isinstance(declared, dict):
        return {}
    return {
        str(action).strip().lower(): str(scope).strip().lower()
        for action, scope in declared.items()
        if str(action).strip() and str(scope).strip()
    }


def action_effect_scope(target: Any, action: Any, skill_scope: Any) -> str:
    """The effect scope of THIS call.

    Falls back to the skill's own scope for an action it never described,
    because an undescribed action is unknown rather than safe.
    """
    declared = declared_action_scopes(target)
    key = str(action or "").strip().lower()
    if not key or key not in declared:
        return str(skill_scope or "unknown").strip().lower()
    return declared[key]


def action_within_scope(target: Any, action: Any, skill_scope: Any, authorised: Any) -> bool:
    """True when this call is no more dangerous than what was authorised."""
    return _rank(action_effect_scope(target, action, skill_scope)) <= _rank(authorised)


def skill_has_action_within(target: Any, skill_scope: Any, authorised: Any) -> bool:
    """True when at least one of the skill's actions is admissible.

    What makes offering a mixed skill defensible: the reader is available and
    the destroyer is refused, rather than the whole skill being withheld
    because part of it is dangerous.
    """
    if _rank(skill_scope) <= _rank(authorised):
        return True
    return any(
        _rank(scope) <= _rank(authorised)
        for scope in declared_action_scopes(target).values()
    )
