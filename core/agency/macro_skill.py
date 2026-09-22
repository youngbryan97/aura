"""A learned macro, exposed as a tool the model can actually call.

``SkillLibrary`` learns sequences of tool calls, persists them, and can now be
searched by meaning. All of which stopped one step short: a retrieved macro was
*described* to the turn and had no way to be invoked. It reached the model as
prose in the context and not as a tool, so the only way to use one was to
re-derive its steps by hand — which is the thing learning it was supposed to
avoid.

This wraps a macro in the canonical ``BaseSkill`` so it can be registered into
the live catalog through ``CapabilityEngine.register_skill``. From that point it
is an ordinary tool: rankable, retrievable, gated, and callable.

The blast radius is derived, not declared
-----------------------------------------
A macro is a sequence of other tools, so its effect scope is the widest scope
among its steps. Asserting one flat scope for every macro would be wrong in both
directions — ``external_io`` over-gates a macro that only reads, and anything
milder under-declares one that writes to disk.

:func:`derive_effect_scope` therefore reads the real scope of each step's tool
out of the catalog policy and takes the maximum. A step whose tool cannot be
resolved collapses the answer to the most restrictive scope, because an unknown
tool is not evidence of a small blast radius.

What this does not do
---------------------
It does not re-gate the steps. Each one still executes through
``tool_orchestrator.execute_tool`` and meets whatever authority that tool
requires. The derived scope governs the *macro* — whether the model may reach
for it at all — and is deliberately no weaker than the sum of its parts.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from core.skills.base_skill import BaseSkill
from core.skills.catalog_policy import VALID_EFFECT_SCOPES

logger = logging.getLogger("Aura.MacroSkill")

__all__ = [
    "MACRO_PREFIX",
    "SCOPE_RANK",
    "MacroSkill",
    "derive_effect_scope",
    "macro_tool_name",
]

#: Registered macros are prefixed so a learned sequence can never take the name
#: of a shipped skill. Shadowing a real capability with a macro somebody's
#: session happened to produce is a supply-chain problem with a friendly face.
MACRO_PREFIX = "macro_"

#: Effect scopes ordered by blast radius, narrowest first.
#:
#: An ordering, not a set of magic numbers: each position is a statement about
#: what the scope can reach, and the only property used is which of two is
#: wider. It mirrors the authority classes in ``core/skills/catalog_policy.py``
#: — observe, then bounded compute, then writes, then the world, then
#: privileged.
SCOPE_RANK: dict[str, int] = {
    "status": 0,
    "read_only": 1,
    "pure_compute": 2,
    "sandboxed_compute": 3,
    "state_mutation": 4,
    "read_write_artifacts": 5,
    "foreground_browser_dialogue": 6,
    "foreground_desktop_control": 7,
    "external_io": 8,
    "privileged_mutation": 9,
}

#: What an unresolvable step is worth. The widest scope there is, because a tool
#: nobody can classify is not thereby harmless.
_UNKNOWN_SCOPE = "privileged_mutation"

_NAME = re.compile(r"[^A-Za-z0-9_-]+")


def macro_tool_name(macro_name: str) -> str:
    """The catalog name for a macro, prefixed and made a legal identifier."""
    cleaned = _NAME.sub("_", str(macro_name or "").strip()).strip("_")
    if not cleaned:
        raise ValueError("macro has no usable name")
    name = f"{MACRO_PREFIX}{cleaned}"
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", name):
        name = name[:64].rstrip("_-")
    return name


def derive_effect_scope(step_tools: list[str], resolve_scope: Any) -> str:
    """The widest effect scope among the macro's steps.

    ``resolve_scope`` takes a tool name and returns its declared scope, or an
    empty string when it cannot be resolved. Injected rather than reached for so
    this stays a pure function of the catalog it is handed.
    """
    if not step_tools:
        # A macro with no steps cannot do anything, and saying so honestly is
        # better than defaulting it into a scope it never exercises.
        return "status"
    widest = "status"
    for tool in step_tools:
        try:
            scope = str(resolve_scope(tool) or "").strip().lower()
        # not a failure: a tool whose scope will not resolve widens nothing, and the
        # status default above stands.
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
            scope = ""
        if scope not in VALID_EFFECT_SCOPES:
            scope = _UNKNOWN_SCOPE
        if SCOPE_RANK.get(scope, SCOPE_RANK[_UNKNOWN_SCOPE]) > SCOPE_RANK[widest]:
            widest = scope
    return widest


class MacroSkill(BaseSkill):
    """One learned macro, callable as a tool.

    Constructed per macro rather than declared in source, so ``name``,
    ``description`` and ``effect_scope`` are instance attributes. The catalog
    reads them off the instance, which is what ``register_skill`` supports.
    """

    def __init__(
        self,
        *,
        macro_name: str,
        description: str,
        parameters: list[str],
        effect_scope: str,
    ) -> None:
        super().__init__()
        self.macro_name = str(macro_name)
        self.name = macro_tool_name(macro_name)
        params = ", ".join(parameters)
        self.description = (
            f"Learned macro: {description}"
            + (f" Parameters: {params}." if params else "")
        ).strip()
        self.parameters = list(parameters)
        if effect_scope not in VALID_EFFECT_SCOPES:
            raise ValueError(f"macro {macro_name!r} derived an unknown effect scope")
        self.effect_scope = effect_scope

    async def execute(
        self, params: dict[str, Any], context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        from core.container import ServiceContainer

        # Arguments first. The check needs nothing but this instance, and
        # reporting "the library is unavailable" to a caller whose real problem
        # is a missing argument sends them to look in the wrong place.
        missing = [p for p in self.parameters if p not in (params or {})]
        if missing:
            # Caught here rather than inside execute_skill so the failure names
            # the tool the model called, not the macro machinery underneath it.
            return {
                "ok": False,
                "error": f"missing required parameter(s): {', '.join(missing)}",
            }

        library = ServiceContainer.get("skill_library", default=None)
        if library is None or not hasattr(library, "execute_skill"):
            return {
                "ok": False,
                "error": "the skill library is unavailable, so this macro cannot run",
            }

        try:
            results = await library.execute_skill(self.macro_name, dict(params or {}))
        except (RuntimeError, TypeError, ValueError, OSError, TimeoutError) as exc:
            # execute_skill already counted the failure and recorded the
            # degradation; this turns it into the result shape a tool caller
            # branches on instead of an exception mid-turn.
            return {"ok": False, "error": str(exc)}

        return {"ok": True, "steps": len(results), "results": results}
