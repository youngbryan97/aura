"""Finite source-consistency diagnostics, with caller-declared witness provenance.

This API checks binding and declared derivation, not observer independence or
the truth of a source interpretation. A dishonest provenance label is not
detectable here. Consistency refers only to the supplied interventions; it
grants neither general correctness nor permission to serve an answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from core.cognition.the_experiment_that_settles_it import what_it_ruled_out, what_to_try
from core.learning.semantic_program_floor import (
    compile_semantic_program_to_floor,
    execute_semantic_floor_program,
)
from core.learning.semantic_program_ir import (
    SemanticProgramIR,
    SemanticValue,
    TokenSpan,
    normalize_semantic_value,
)

MAX_CANDIDATES = 16
MAX_INTERVENTIONS = 32
MAX_WITNESSES = 128
MAX_EXECUTION_FUEL = 100_000


def _name(value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("a non-empty name or reference is required")


def _sha256(value: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError("a lowercase SHA256 digest is required")


class ValueKind(StrEnum):
    INTEGER = "integer"
    INTEGER_SEQUENCE = "integer_sequence"


class WitnessDerivation(StrEnum):
    SOURCE_OBSERVATION = "source_observation"
    CANDIDATE_DERIVED = "candidate_derived"


class DiagnosticStatus(StrEnum):
    CONTRADICTED = "contradicted"
    UNRESOLVED = "unresolved"
    CONSISTENT_ON_DECLARED_SCOPE = "consistent_on_declared_scope"


@dataclass(frozen=True, slots=True)
class OperandRole:
    """A caller's source-role assertion, ordered by the IR input registers."""

    name: str
    span: TokenSpan
    kind: ValueKind

    def __post_init__(self) -> None:
        _name(self.name)
        if not isinstance(self.span, TokenSpan) or not isinstance(self.kind, ValueKind):
            raise ValueError("operand roles require a token span and a typed value kind")


@dataclass(frozen=True, slots=True)
class Intervention:
    """Replacement input values in the declared operand-role order."""

    name: str
    inputs: tuple[SemanticValue, ...]

    def __post_init__(self) -> None:
        _name(self.name)
        if not isinstance(self.inputs, tuple) or not self.inputs:
            raise ValueError("intervention inputs must be a non-empty tuple")
        for value in self.inputs:
            if type(value) not in (int, tuple):
                raise ValueError("intervention values must be immutable exact values")
            normalize_semantic_value(value)


@dataclass(frozen=True, slots=True)
class SourceScope:
    """Exact finite scope, including the caller's meaning-preservation assertion.

    ``interpretation`` states how replacing inputs preserves the source's
    operations and role bindings. This adapter cannot validate that assertion.
    """

    source_text_sha256: str
    source_token_ids: tuple[int, ...]
    roles: tuple[OperandRole, ...]
    interpretation: str
    interventions: tuple[Intervention, ...]

    def __post_init__(self) -> None:
        _sha256(self.source_text_sha256)
        _name(self.interpretation)
        if (
            not isinstance(self.source_token_ids, tuple)
            or not self.source_token_ids
            or any(type(t) is not int or t < 0 for t in self.source_token_ids)
        ):
            raise ValueError("scope requires the source token sequence used by its role spans")
        if (
            not isinstance(self.roles, tuple)
            or not self.roles
            or not all(isinstance(r, OperandRole) for r in self.roles)
        ):
            raise ValueError("scope requires ordered operand roles")
        if len({r.name for r in self.roles}) != len(self.roles):
            raise ValueError("operand role names must be unique")
        for role in self.roles:
            role.span.validate_bound(len(self.source_token_ids))
        spans = sorted(r.span for r in self.roles)
        if any(left.end > right.start for left, right in zip(spans, spans[1:], strict=False)):
            raise ValueError("operand role spans must not overlap")
        if not isinstance(self.interventions, tuple) or len(self.interventions) > MAX_INTERVENTIONS:
            raise ValueError("scope exceeds the finite intervention limit")
        if not all(isinstance(i, Intervention) for i in self.interventions):
            raise ValueError("scope requires typed interventions")
        if len({i.name for i in self.interventions}) != len(self.interventions):
            raise ValueError("intervention names must be unique")
        for intervention in self.interventions:
            if len(intervention.inputs) != len(self.roles):
                raise ValueError("intervention arity differs from source roles")
            for role, value in zip(self.roles, intervention.inputs, strict=True):
                wanted = int if role.kind is ValueKind.INTEGER else tuple
                if type(value) is not wanted:
                    raise ValueError("intervention type differs from source role")


@dataclass(frozen=True, slots=True)
class WitnessProvenance:
    """Caller declarations only; no authentication is implied by these fields."""

    observer: str
    reference: str
    derivation: WitnessDerivation
    candidate_dependencies: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _name(self.observer)
        _name(self.reference)
        if not isinstance(self.derivation, WitnessDerivation):
            raise ValueError("witness derivation must be explicit")
        if not isinstance(self.candidate_dependencies, tuple):
            raise ValueError("candidate dependencies must be immutable")
        for digest in self.candidate_dependencies:
            _sha256(digest)


@dataclass(frozen=True, slots=True)
class SourceWitness:
    """An observation bound to an entire scope and one intervention in it.

    None means the observer supplied no successful observation, including when
    observation failed. Exceptions are never converted to comparable answers.
    """

    scope: SourceScope
    intervention: Intervention
    observed: SemanticValue | None
    provenance: WitnessProvenance

    def __post_init__(self) -> None:
        if (
            not isinstance(self.scope, SourceScope)
            or self.intervention not in self.scope.interventions
        ):
            raise ValueError("witness intervention is outside its source scope")
        if not isinstance(self.provenance, WitnessProvenance):
            raise ValueError("witness requires declared provenance")
        if self.observed is not None:
            if type(self.observed) not in (int, tuple):
                raise ValueError("witness observation must be an immutable exact value")
            normalize_semantic_value(self.observed)


@dataclass(frozen=True, slots=True)
class ProgramCandidate:
    name: str
    ir: SemanticProgramIR

    def __post_init__(self) -> None:
        _name(self.name)
        if not isinstance(self.ir, SemanticProgramIR):
            raise ValueError("candidate requires validated semantic IR")


@dataclass(frozen=True, slots=True)
class Prediction:
    candidate: str
    value: SemanticValue | None
    error: str = ""


@dataclass(frozen=True, slots=True)
class ProbeRecord:
    intervention: Intervention
    witnesses: tuple[SourceWitness, ...]
    predictions: tuple[Prediction, ...]
    contradicted: tuple[str, ...]
    observation_issue: str = ""


@dataclass(frozen=True, slots=True)
class CandidateAssessment:
    candidate: ProgramCandidate
    status: DiagnosticStatus
    matched: tuple[str, ...]
    counterexamples: tuple[str, ...]
    silent: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SourceArbitrationDiagnostic:
    scope: SourceScope
    assessments: tuple[CandidateAssessment, ...]
    probes: tuple[ProbeRecord, ...]
    untested: tuple[str, ...]
    prediction_count: int
    execution_fuel: int
    provenance_authenticated: bool = field(default=False, init=False)
    serving_authority: bool = field(default=False, init=False)


def diagnose_source_programs(
    *,
    scope: SourceScope,
    candidates: tuple[ProgramCandidate, ...],
    witnesses: tuple[SourceWitness, ...] = (),
    max_probes: int = MAX_INTERVENTIONS,
    execution_fuel: int = 10_000,
) -> SourceArbitrationDiagnostic:
    """Compare candidate predictions with supplied source observations.

    The chooser can predict all remaining interventions to choose one. Results
    are cached, bounding executions by candidates times interventions, each
    with the supplied floor fuel. ``max_probes`` bounds observation comparisons,
    not predictions. No witness provider, model, registry, or live runtime is
    called. Fuel bounds floor reductions, not integer bit complexity or wall
    time; callers must also bound their input sizes when running this diagnostic.
    """
    if not isinstance(scope, SourceScope):
        raise ValueError("diagnostic requires a typed source scope")
    if not isinstance(candidates, tuple) or not 1 <= len(candidates) <= MAX_CANDIDATES:
        raise ValueError("diagnostic requires a bounded non-empty candidate tuple")
    if not all(isinstance(c, ProgramCandidate) for c in candidates):
        raise ValueError("diagnostic requires typed program candidates")
    if len({c.name for c in candidates}) != len(candidates):
        raise ValueError("candidate names must be unique")
    if type(max_probes) is not int or not 0 <= max_probes <= MAX_INTERVENTIONS:
        raise ValueError("invalid probe budget")
    if type(execution_fuel) is not int or not 1 <= execution_fuel <= MAX_EXECUTION_FUEL:
        raise ValueError("invalid execution fuel")
    for candidate in candidates:
        if (
            candidate.ir.source_text_sha256 != scope.source_text_sha256
            or candidate.ir.source_token_ids != scope.source_token_ids
            or candidate.ir.input_spans != tuple(r.span for r in scope.roles)
        ):
            raise ValueError("candidate source or ordered role spans differ from scope")
    if not isinstance(witnesses, tuple) or len(witnesses) > MAX_WITNESSES:
        raise ValueError("witnesses must be a bounded tuple")
    by_act: dict[str, list[SourceWitness]] = {i.name: [] for i in scope.interventions}
    for witness in witnesses:
        if not isinstance(witness, SourceWitness) or witness.scope != scope:
            raise ValueError("witness source, roles, or scope differ from diagnostic")
        if (
            witness.provenance.derivation is WitnessDerivation.CANDIDATE_DERIVED
            or witness.provenance.candidate_dependencies
        ):
            raise ValueError("candidate-derived witnesses cannot arbitrate source meaning")
        by_act[witness.intervention.name].append(witness)

    cache: dict[tuple[str, str], Prediction] = {}

    def predicts(candidate: ProgramCandidate, act: Intervention) -> SemanticValue | None:
        key = (candidate.name, act.name)
        if key not in cache:
            try:
                program = compile_semantic_program_to_floor(candidate.ir, act.inputs)
                value = execute_semantic_floor_program(program, fuel=execution_fuel).result
                value = None if value is None else normalize_semantic_value(value)
                cache[key] = Prediction(candidate.name, value)
            except Exception as exc:  # noqa: BLE001 - failed predictions remain unresolved
                cache[key] = Prediction(candidate.name, None, type(exc).__name__)
        return cache[key].value

    field_by_name = {c.name: c for c in candidates}
    standing = dict(field_by_name)
    pending = list(scope.interventions)
    probes: list[ProbeRecord] = []
    matched: dict[str, list[str]] = {c.name: [] for c in candidates}
    wrong: dict[str, list[str]] = {c.name: [] for c in candidates}
    silent: dict[str, list[str]] = {c.name: [] for c in candidates}
    while pending and len(probes) < max_probes:
        choice = what_to_try(standing, pending, predicts=predicts, plausibility=lambda _n, _c: 1.0)
        # With no discriminating act, still compare observations for coverage.
        act = choice.do if choice is not None else pending[0]
        pending.remove(act)
        observed = tuple(by_act[act.name])
        values = {w.observed for w in observed if w.observed is not None}
        issue = ""
        if not values:
            issue = "missing_observation"
        elif len(values) != 1:
            issue = "conflicting_observations"
        elif any(w.observed is None for w in observed):
            issue = "incomplete_observations"
        for candidate in candidates:
            predicts(candidate, act)
            if cache[(candidate.name, act.name)].value is None:
                silent[candidate.name].append(act.name)
        contradicted: tuple[str, ...] = ()
        if not issue:
            saw = next(iter(values))
            survivors = what_it_ruled_out(field_by_name, act, saw, predicts=predicts)
            contradicted = tuple(name for name in field_by_name if name not in survivors)
            standing = {name: c for name, c in standing.items() if name in survivors}
            for name in field_by_name:
                if name in contradicted:
                    wrong[name].append(act.name)
                elif cache[(name, act.name)].value is not None:
                    matched[name].append(act.name)
        probes.append(
            ProbeRecord(
                act,
                observed,
                tuple(cache[(c.name, act.name)] for c in candidates),
                contradicted,
                issue,
            )
        )

    assessments = []
    for candidate in candidates:
        name = candidate.name
        status = DiagnosticStatus.UNRESOLVED
        if wrong[name]:
            status = DiagnosticStatus.CONTRADICTED
        elif scope.interventions and len(matched[name]) == len(scope.interventions):
            status = DiagnosticStatus.CONSISTENT_ON_DECLARED_SCOPE
        assessments.append(
            CandidateAssessment(
                candidate, status, tuple(matched[name]), tuple(wrong[name]), tuple(silent[name])
            )
        )
    return SourceArbitrationDiagnostic(
        scope,
        tuple(assessments),
        tuple(probes),
        tuple(i.name for i in pending),
        len(cache),
        execution_fuel,
    )
