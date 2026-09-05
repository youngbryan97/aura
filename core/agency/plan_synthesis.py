"""Building a plan out of things she can really do.

A plan is a list of steps, and a step carries a callable. That is why
planning has always arrived here as an injected lambda: a language model
cannot write a callable, so nothing could turn an intention into a plan.

The model does not have to. It has to name an action and its arguments; the
architecture does the rest. Every registered skill already declares a name, a
description, and a JSON schema for its input, so the actions available are a
typed vocabulary — closed, validated, and the same one tool dispatch uses.
This module reads that vocabulary, checks a proposed call against it, and
compiles what survives into executable steps.

What this buys is composition. Tool dispatch runs one named skill against one
message. A synthesized plan is any sequence of them, chosen for a goal, with
each step's arguments validated before anything runs — and a step that fails
validation is reported as the error it is, so the next attempt is told what
was wrong rather than asked again in different words.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger(__name__)

#: Actions that change the world outside her, which a plan may only include
#: when the caller has said so.
WORLD_CHANGING = {"environment_action", "external_action", "filesystem_write"}
#: How many times a plan is rebuilt when the arguments do not validate.
COMPILE_ATTEMPTS = 3


@dataclass(frozen=True)
class ActionSpec:
    """One thing she can really do, and the arguments it takes."""

    name: str
    description: str
    schema: Mapping[str, Any] = field(default_factory=dict)
    effect_scope: str = "unknown"
    requires_approval: bool = False

    def arguments(self) -> list[str]:
        properties = self.schema.get("properties") if isinstance(self.schema, Mapping) else None
        return sorted(properties) if isinstance(properties, Mapping) else []

    def required(self) -> list[str]:
        required = self.schema.get("required") if isinstance(self.schema, Mapping) else None
        return list(required) if isinstance(required, list) else []

    def as_line(self) -> str:
        args = ", ".join(self.arguments())
        detail = f" — takes: {args}" if args else " — takes no arguments"
        return f"{self.name}: {self.description}{detail}"


@dataclass(frozen=True)
class ProposedCall:
    """A named action with arguments, before anything has checked it."""

    action: str
    args: Mapping[str, Any] = field(default_factory=dict)
    because: str = ""


@dataclass
class SynthesizedPlan:
    """Steps that can run, the calls that could not, and why."""

    goal: str
    steps: list[Any] = field(default_factory=list)
    calls: list[ProposedCall] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    attempts: int = 0

    @property
    def usable(self) -> bool:
        return bool(self.steps)

    def narrate(self) -> str:
        if not self.steps:
            return f"I could not build a plan for {self.goal}: " + "; ".join(self.rejected)
        return " then ".join(str(getattr(step, "name", "")) for step in self.steps)


def action_space(*, engine: Any = None, allow_world_changing: bool = False) -> list[ActionSpec]:
    """Everything she can do right now, as a vocabulary a plan can be built from."""
    if engine is None:
        engine = _engine()
    if engine is None:
        return []
    specs: list[ActionSpec] = []
    try:
        catalog = dict(getattr(engine, "skills", {}) or {})
    except (AttributeError, RuntimeError, TypeError) as exc:
        record_degradation("plan_synthesis", exc, action="read the action vocabulary")
        return []
    for name, meta in catalog.items():
        if getattr(meta, "enabled", True) is False:
            continue
        scope = str(getattr(meta, "effect_scope", "unknown") or "unknown")
        if scope in WORLD_CHANGING and not allow_world_changing:
            continue
        try:
            schema = meta.schema_def() if hasattr(meta, "schema_def") else {}
        except (AttributeError, RuntimeError, TypeError, ValueError):
            schema = {}
        specs.append(
            ActionSpec(
                name=str(name),
                description=str(getattr(meta, "description", "") or ""),
                schema=schema,
                effect_scope=scope,
                requires_approval=bool(getattr(getattr(meta, "requirements", None), "requires_approval", False)),
            )
        )
    specs.sort(key=lambda spec: spec.name)
    return specs


def _engine() -> Any:
    try:
        from core.container import get_container  # noqa: PLC0415
        from core.exceptions import ContainerError  # noqa: PLC0415

        try:
            return get_container().get("capability_engine")
        except (ContainerError, KeyError):
            # Nothing registered is a state, not a failure: the caller is told
            # she has no actions to plan with.
            return None
    except (ImportError, AttributeError, RuntimeError) as exc:
        record_degradation("plan_synthesis", exc, action="reach the capability engine")
        return None


def read_calls(reply: str) -> list[ProposedCall]:
    """Read proposed calls out of a reply, ignoring anything that is not one."""
    from core.utils.json_utils import extract_json  # noqa: PLC0415

    payload: Any = None
    try:
        payload = json.loads(reply)
    except (TypeError, ValueError):
        try:
            # extract_json falls back to literal_eval, which raises SyntaxError
            # on ordinary prose. A reply that is not a plan is not an error.
            payload = extract_json(reply)
        except (TypeError, ValueError, AttributeError, SyntaxError):
            payload = None
    if isinstance(payload, Mapping):
        payload = payload.get("steps") or payload.get("plan") or payload.get("actions") or [payload]
    if not isinstance(payload, list):
        return []
    calls: list[ProposedCall] = []
    for item in payload:
        if not isinstance(item, Mapping):
            continue
        action = str(item.get("action") or item.get("name") or item.get("skill") or "").strip()
        if not action:
            continue
        args = item.get("args") or item.get("arguments") or item.get("params") or {}
        calls.append(
            ProposedCall(
                action=action,
                args=dict(args) if isinstance(args, Mapping) else {},
                because=str(item.get("because") or item.get("why") or "").strip(),
            )
        )
    return calls


def check_call(call: ProposedCall, space: Sequence[ActionSpec]) -> tuple[ActionSpec | None, list[str]]:
    """Match a proposed call to a real action and check its arguments.

    Returns the action and every problem found. A problem is a fact about the
    call, phrased so the next attempt can be told what was wrong.
    """
    by_name = {spec.name: spec for spec in space}
    spec = by_name.get(call.action)
    if spec is None:
        return None, [f"{call.action!r} is not something she can do"]
    problems: list[str] = []
    known = set(spec.arguments())
    for missing in spec.required():
        if missing not in call.args:
            problems.append(f"{spec.name} needs {missing!r} and it was not given")
    if known:
        for extra in sorted(set(call.args) - known):
            problems.append(f"{spec.name} takes no argument called {extra!r}")
    return spec, problems


def compile_step(spec: ActionSpec, call: ProposedCall, *, engine: Any = None, context: Mapping[str, Any] | None = None) -> Any:
    """Turn a checked call into a step that runs it.

    This is the part a model cannot do and does not need to. The callable is
    built here, bound to the same capability engine tool dispatch uses, so a
    synthesized plan runs through every check a directly requested skill does.
    """
    from core.skills.fluid_executor import Step  # noqa: PLC0415

    args = dict(call.args)
    ctx = dict(context or {})
    ctx.setdefault("requested_via", "plan_synthesis")

    async def run() -> bool:
        target = engine if engine is not None else _engine()
        if target is None:
            raise RuntimeError("no capability engine is available")
        result = await target.execute(spec.name, args, ctx)
        if isinstance(result, Mapping):
            if "ok" in result:
                return bool(result.get("ok"))
            if "success" in result:
                return bool(result.get("success"))
            return "error" not in result
        return bool(result)

    return Step(name=f"{spec.name}({', '.join(sorted(args))})", action=run, approach="plan_synthesis")


def _objective(goal: str, space: Sequence[ActionSpec]) -> str:
    return (
        f"Build a plan that reaches this goal: {goal}. "
        "Answer with a JSON list of steps. Each step is an object with "
        '"action" naming one available action, "args" giving its arguments, '
        'and "because" saying in one sentence why that step is there.'
    )


async def synthesize_plan(
    goal: str,
    *,
    think: Any,
    engine: Any = None,
    space: Sequence[ActionSpec] | None = None,
    allow_world_changing: bool = False,
    context: Mapping[str, Any] | None = None,
    attempts: int = COMPILE_ATTEMPTS,
    evidence: Sequence[str] = (),
) -> SynthesizedPlan:
    """Build an executable plan for ``goal`` out of the actions available.

    A call whose arguments do not check out is not retried by asking again in
    other words. The specific problem goes back as a fact — the argument that
    was missing, the action that does not exist — the same way a compiler
    reports an error, and the next attempt has something to work from.
    """
    vocabulary = list(space) if space is not None else action_space(
        engine=engine, allow_world_changing=allow_world_changing
    )
    plan = SynthesizedPlan(goal=goal)
    if not vocabulary:
        plan.rejected.append("she has no actions available to plan with")
        return plan

    facts = [f"Available action — {spec.as_line()}" for spec in vocabulary]
    facts.extend(evidence)
    problems: list[str] = []

    for attempt in range(1, max(1, attempts) + 1):
        plan.attempts = attempt
        try:
            reply = await think(_objective(goal, vocabulary), facts + problems)
        except (RuntimeError, AttributeError, TypeError, ValueError, TimeoutError) as exc:
            record_degradation("plan_synthesis", exc, action="reason about a plan")
            plan.rejected.append(f"her reasoning could not be reached ({type(exc).__name__})")
            return plan

        calls = read_calls(reply or "")
        if not calls:
            problems = ["The last answer named no action. Answer with the JSON list of steps."]
            plan.rejected.append("she named no action")
            continue

        steps: list[Any] = []
        found: list[str] = []
        for call in calls:
            spec, trouble = check_call(call, vocabulary)
            if spec is None or trouble:
                found.extend(trouble)
                continue
            steps.append(compile_step(spec, call, engine=engine, context=context))
        if steps and not found:
            plan.steps = steps
            plan.calls = calls
            plan.rejected = []
            return plan
        problems = found or ["No step could be built from the last answer."]
        plan.rejected = list(found)
        plan.steps = steps
        plan.calls = calls

    return plan
