"""core/cognition/procedure_adapters.py — the eight learners, priced together.

:mod:`core.cognition.procedure` defines the currency. This is where the
existing learners are converted into it, and the conversions are the
interesting part: each learner's value function says something the others do
not, and flattening them all to one float would throw that away.

* A **chunk** measures ``cost_saved_per_use`` from the substate that produced
  it, in seconds. That is already the right unit for
  ``value_when_it_works``, and its ``match_cost`` is already the utility
  problem's other term, so the conversion is close to an identity.

* A **generalized rule** has a Wilson lower bound over pooled derivation and
  post-promotion evidence. That is a *conservative* ``p_success``, and it is
  kept conservative: the rule's optimistic ratio is not used, because the whole
  point of the Wilson bound is that a rule promoted on twelve episodes should
  not be demoted by its first three correct outcomes, nor promoted by them.

* A **learned skill** reports reliability with a 0.5 prior and no cost model at
  all, so the conversion has to supply one. It supplies a *measured* one —
  step count times a per-step cost the caller passes in — rather than inventing
  a constant, and a skill with no measured value converts to net zero, which
  means it neither wins a match nor gets retired for losing one.

The direction matters. Nothing here writes back into the source learners: they
keep their own stores, their own arithmetic and their own tests. The registry
is a view that lets them compete, and a procedure retired in the registry is
not deleted from its learner — it is de-prioritised in the one place that
ranks across learners.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from core.cognition.procedure import (
    Backend,
    Effect,
    Origin,
    Precondition,
    Procedure,
    ProceduralValue,
    ProcedureRegistry,
    Reversibility,
    Signature,
    get_procedure_registry,
)

__all__ = [
    "from_chunk",
    "from_generalized_rule",
    "from_learned_skill",
    "from_doing",
    "from_tool_schema",
    "ingest_all",
    "install_the_learners",
    "whatever_the_learners_hold",
]


def from_chunk(chunk: Any, *, registry: ProcedureRegistry | None = None) -> Procedure:
    """A compiled impasse resolution, in the common currency.

    ``expected_value`` already IS ``p_correct · cost_saved − match_cost``, so
    the terms transfer one for one and the registry's ``net`` reproduces the
    chunk's own number. That is the check that the currency is compatible with
    the arithmetic it generalises.
    """
    registry = registry or get_procedure_registry()
    return registry.register(
        f"chunk:{chunk.signature}",
        Backend.CHUNK,
        Signature(
            preconditions=(Precondition(key=f"situation:{chunk.signature}"),),
            effects=(Effect(key="resolution", value=chunk.resolution),),
        ),
        program=chunk.resolution,
        value=ProceduralValue(
            p_success=chunk.p_correct,
            value_when_it_works=chunk.cost_saved_per_use,
            match_cost=chunk.match_cost,
            uses=chunk.uses,
            successes=chunk.correct,
        ),
        origin=Origin(
            learner="core.cognition.impasse",
            impasse_type=str(chunk.impasse_type),
            support_keys=(f"situation:{chunk.signature}",),
        ),
        reversibility=Reversibility.REVERSIBLE,
    )


def from_generalized_rule(
    rule: Any,
    *,
    value_when_it_works: float,
    match_cost: float = 0.0,
    registry: ProcedureRegistry | None = None,
) -> Procedure:
    """A promoted procedural rule, priced under its own Wilson bound.

    ``value_when_it_works`` has to come from the caller because the rule store
    never measured one — it measured whether the rule is RIGHT, not what being
    right is worth. Asking for it here is better than inventing it: a caller
    that does not know passes 0.0 and gets a procedure that competes on nothing,
    which is the truthful ranking for a rule whose benefit was never measured.
    """
    registry = registry or get_procedure_registry()
    conditions = tuple(sorted(str(c) for c in rule.conditions))
    successes = rule.supporting + rule.correct
    trials = successes + rule.contradicting + rule.incorrect
    return registry.register(
        f"rule:{rule.resolution}",
        Backend.GENERALIZED_RULE,
        Signature(
            preconditions=tuple(Precondition(key=c) for c in conditions),
            effects=(Effect(key="resolution", value=rule.resolution),),
        ),
        program=rule.resolution,
        value=ProceduralValue(
            p_success=rule.confidence,  # the conservative reading, deliberately
            value_when_it_works=value_when_it_works,
            match_cost=match_cost,
            uses=trials,
            successes=successes,
        ),
        origin=Origin(
            learner="core.cognition.procedural_generalization",
            support_keys=conditions,
            rejected_conditions=tuple(str(f) for f in rule.lesioned),
        ),
        reversibility=Reversibility.REVERSIBLE,
    )


def from_learned_skill(
    skill: Any,
    *,
    seconds_per_step: float,
    match_cost: float = 0.0,
    registry: ProcedureRegistry | None = None,
) -> Procedure:
    """A macro of tool calls, given the cost model it never had.

    ``seconds_per_step`` must be measured by the caller. A skill with no
    measured per-step cost converts at 0.0 and lands at net zero: it will not
    beat a chunk that measured its saving, and it will not be retired for
    losing, which is the correct treatment of a procedure nobody has timed.
    """
    registry = registry or get_procedure_registry()
    trials = skill.successes + skill.failures
    return registry.register(
        f"skill:{skill.name}",
        Backend.MACRO,
        Signature(
            preconditions=tuple(Precondition(key=p) for p in skill.parameters),
            effects=(Effect(key=f"skill:{skill.name}:done"),),
        ),
        program=list(skill.steps),
        value=ProceduralValue(
            p_success=skill.reliability,
            value_when_it_works=seconds_per_step * len(skill.steps),
            match_cost=match_cost,
            uses=trials,
            successes=skill.successes,
        ),
        origin=Origin(learner="core.agency.skill_library"),
        reversibility=Reversibility.UNKNOWN,
    )


def from_doing(
    doing: Any,
    *,
    name: str,
    reads: Sequence[str] = (),
    writes: Sequence[str] = (),
    value_when_it_works: float = 0.0,
    p_success: float = 0.5,
    registry: ProcedureRegistry | None = None,
) -> Procedure:
    """A composed action from an_action_she_composed, as a typed procedure.

    A ``Doing`` is an AST over world operations and carries no signature of its
    own; the caller supplies what it reads and writes. That is honest rather
    than convenient — inferring the signature from the AST would need the world
    it was composed against, and a wrong signature composes into a wrong plan.
    """
    registry = registry or get_procedure_registry()
    return registry.register(
        name,
        Backend.DOING,
        Signature(
            preconditions=tuple(Precondition(key=r) for r in reads),
            effects=tuple(Effect(key=w) for w in writes),
        ),
        program=doing,
        value=ProceduralValue(p_success=p_success, value_when_it_works=value_when_it_works),
        origin=Origin(learner="core.cognition.an_action_she_composed"),
        reversibility=Reversibility.UNKNOWN,
    )


def from_tool_schema(
    name: str,
    *,
    requires: Sequence[str] = (),
    produces: Sequence[str] = (),
    reversibility: Reversibility = Reversibility.UNKNOWN,
    risk_cost: float = 0.0,
    registry: ProcedureRegistry | None = None,
) -> Procedure:
    """A tool, so a learned procedure can compose with one it did not learn."""
    registry = registry or get_procedure_registry()
    return registry.register(
        f"tool:{name}",
        Backend.TOOL,
        Signature(
            preconditions=tuple(Precondition(key=r) for r in requires),
            effects=tuple(Effect(key=p) for p in produces),
        ),
        program=name,
        value=ProceduralValue(p_success=1.0, risk_cost=risk_cost),
        reversibility=reversibility,
    )


def ingest_all(
    *,
    chunks: Iterable[Any] = (),
    rules: Iterable[Any] = (),
    skills: Iterable[Any] = (),
    rule_value: float = 0.0,
    seconds_per_step: float = 0.0,
    registry: ProcedureRegistry | None = None,
) -> dict[str, Any]:
    """Pull every learner's store into the registry once. Returns what landed."""
    registry = registry or get_procedure_registry()
    landed: dict[str, list[str]] = {"chunk": [], "rule": [], "skill": []}
    for chunk in chunks:
        landed["chunk"].append(from_chunk(chunk, registry=registry).procedure_id)
    for rule in rules:
        landed["rule"].append(
            from_generalized_rule(rule, value_when_it_works=rule_value, registry=registry).procedure_id
        )
    for skill in skills:
        landed["skill"].append(
            from_learned_skill(skill, seconds_per_step=seconds_per_step, registry=registry).procedure_id
        )
    return {"landed": {k: len(v) for k, v in landed.items()}, "ids": landed}


def whatever_the_learners_hold(
    *,
    registry: ProcedureRegistry | None = None,
    seconds_per_step: float | None = None,
    rule_value: float | None = None,
) -> dict[str, Any]:
    """Find the three stores and price everything in them together.

    ``ingest_all`` takes the stores as arguments, so somebody has to go and
    get them. Nobody did: this module had no importer anywhere in production,
    while the claim ladder cited it as the WIRED evidence that procedures from
    different learners compete under one value. They could have. They never
    were asked to.

    The two prices are measured rather than assumed, and where a measurement
    is missing the caller's default of zero stands — a learner that never
    timed itself competes on nothing, which is the truthful ranking for a
    procedure nobody has timed.

    A store that will not import or will not answer is skipped and named in
    the result, because a partial ranking that says which learners are in it
    is worth more than one that quietly ranks two of three.
    """
    registry = registry or get_procedure_registry()
    held: dict[str, Any] = {"chunks": (), "rules": (), "skills": ()}
    missing: list[str] = []

    try:
        from core.cognition.impasse import get_impasse_learner

        # The learner holds the store; the store is what lists the chunks.
        held["chunks"] = tuple(get_impasse_learner()._store.chunks())  # noqa: SLF001
    except Exception as exc:  # noqa: BLE001 - a store that is not there is data
        missing.append(f"chunks ({type(exc).__name__})")

    try:
        from core.cognition.procedural_generalization import (
            get_procedural_generalizer,
        )

        held["rules"] = tuple(get_procedural_generalizer().rules())
    except Exception as exc:  # noqa: BLE001
        missing.append(f"rules ({type(exc).__name__})")

    try:
        from core.container import ServiceContainer

        library = ServiceContainer.get("skill_library", default=None)
        held["skills"] = tuple(library.skills.values()) if library else ()
        if library is None:
            missing.append("skills (no skill_library in the container)")
    except Exception as exc:  # noqa: BLE001
        missing.append(f"skills ({type(exc).__name__})")

    landed = ingest_all(
        chunks=held["chunks"],
        rules=held["rules"],
        skills=held["skills"],
        rule_value=_a_rule_is_worth(held["chunks"]) if rule_value is None else rule_value,
        seconds_per_step=(
            _what_a_step_costs() if seconds_per_step is None else seconds_per_step
        ),
        registry=registry,
    )
    landed["missing"] = tuple(missing)
    return landed


def _a_rule_is_worth(chunks: Sequence[Any]) -> float:
    """What being right is worth, from the one learner that measured it.

    The rule store measured whether a rule is right and never what being right
    saves. Rather than invent a number, take the median saving the chunk
    learner measured on the same kind of work. With no chunks there is no
    measurement, and zero is the honest answer.
    """
    from statistics import median

    saved = [
        float(one.cost_saved_per_use)
        for one in chunks
        if getattr(one, "cost_saved_per_use", None) is not None
    ]
    return median(saved) if saved else 0.0


def _what_a_step_costs() -> float:
    """Seconds per tool call, read from what tool calls have actually taken."""
    try:
        from core.cognition.procedure import get_procedure_registry as _reg

        timings = [
            one.value.value_when_it_works / max(1, len(one.program or ()))
            for one in _reg().all()
            if one.backend is Backend.MACRO and one.value.value_when_it_works > 0
        ]
        return sum(timings) / len(timings) if timings else 0.0
    except Exception:  # noqa: BLE001 - no measurement is 0.0, not a guess
        return 0.0


def install_the_learners(registry: ProcedureRegistry | None = None) -> bool:
    """Tell the registry where the other learners keep their procedures.

    Without this the registry holds only what is registered through it
    directly, which in production was the semantic-program path and nothing
    else — so "backends compete directly" described the arithmetic and not
    the running system. Safe to call again; the last caller wins.
    """
    (registry or get_procedure_registry()).keep_current_with(
        lambda: whatever_the_learners_hold(registry=registry)
    )
    return True
