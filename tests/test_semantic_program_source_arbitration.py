"""Finite semantic diagnostics never substitute execution for source evidence."""

from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace

import pytest

from core.learning import semantic_program_source_arbitration as arb
from core.learning.semantic_program_ir import SemanticIRInstruction, SemanticProgramIR, TokenSpan


def _candidate(name="add", instructions=None, inputs=2):
    instructions = instructions or ((name, (0, 1)),)
    ir = SemanticProgramIR(
        source_token_ids=tuple(range(inputs + len(instructions) + 3)),
        source_text_sha256=sha256(b"source contract fixture").hexdigest(),
        input_spans=tuple(TokenSpan(i, i + 1) for i in range(inputs)),
        instructions=tuple(
            SemanticIRInstruction(
                op=op,
                args=args,
                operation_span=TokenSpan(inputs + i, inputs + i + 1),
                argument_spans=tuple(TokenSpan(a, a + 1) for a in args),
                depends_on=tuple(sorted({a - inputs for a in args if a >= inputs})),
            )
            for i, (op, args) in enumerate(instructions)
        ),
        report_value=inputs + len(instructions) - 1,
        model_basis_receipt_sha256="1" * 64,
        transducer_receipt_sha256="2" * 64,
    )
    return arb.ProgramCandidate(name, ir)


def _scope(*values, candidate=None):
    candidate = candidate or _candidate()
    values = values or ((2, 2), (2, 3))
    return arb.SourceScope(
        candidate.ir.source_text_sha256,
        candidate.ir.source_token_ids,
        tuple(
            arb.OperandRole(
                f"operand:{i}",
                span,
                arb.ValueKind.INTEGER
                if type(values[0][i]) is int
                else arb.ValueKind.INTEGER_SEQUENCE,
            )
            for i, span in enumerate(candidate.ir.input_spans)
        ),
        "Replace operand values while preserving the source's operations and roles.",
        tuple(arb.Intervention(f"case:{i}", v) for i, v in enumerate(values)),
    )


def _witnesses(scope, *answers):
    provenance = arb.WitnessProvenance(
        "fixture author",
        "independently authored contract",
        arb.WitnessDerivation.SOURCE_OBSERVATION,
    )
    return tuple(
        arb.SourceWitness(scope, act, answer, provenance)
        for act, answer in zip(scope.interventions, answers, strict=True)
    )


def _run(scope, candidates=None, witnesses=(), **kwargs):
    return arb.diagnose_source_programs(
        scope=scope,
        candidates=candidates or (_candidate(), _candidate("mul")),
        witnesses=witnesses,
        **kwargs,
    )


def test_discriminating_act_rejects_executable_wrong_operation():
    scope = _scope()
    result = _run(scope, witnesses=_witnesses(scope, 4, 5))
    assert result.probes[0].intervention.inputs == (2, 3)
    good, bad = result.assessments
    assert good.status is arb.DiagnosticStatus.CONSISTENT_ON_DECLARED_SCOPE
    assert bad.status is arb.DiagnosticStatus.CONTRADICTED
    assert bad.counterexamples == ("case:1",)
    assert result.probes[0].predictions == (arb.Prediction("add", 5), arb.Prediction("mul", 6))
    assert result.prediction_count == 4
    assert result.serving_authority is False
    assert result.provenance_authenticated is False


def test_adapter_calls_existing_chooser_and_eliminator(monkeypatch):
    calls = []
    for name in ("what_to_try", "what_it_ruled_out"):
        original = getattr(arb, name)

        def wrapped(*args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(arb, name, wrapped)
    scope = _scope()
    _run(scope, witnesses=_witnesses(scope, 4, 5))
    assert calls.count("what_to_try") == 2
    assert calls.count("what_it_ruled_out") == 2


def test_argument_order_and_branch_binding_are_falsifiable():
    good = _candidate("good", (("sub", (0, 1)), ("mul", (3, 2))), inputs=3)
    reversed_args = _candidate("reversed", (("sub", (1, 0)), ("mul", (3, 2))), inputs=3)
    wrong_branch = _candidate("branch", (("sub", (0, 2)), ("mul", (3, 1))), inputs=3)
    scope = _scope((7, 2, 3), candidate=good)
    result = _run(scope, (good, reversed_args, wrong_branch), _witnesses(scope, 15))
    assert [a.status for a in result.assessments] == [
        arb.DiagnosticStatus.CONSISTENT_ON_DECLARED_SCOPE,
        arb.DiagnosticStatus.CONTRADICTED,
        arb.DiagnosticStatus.CONTRADICTED,
    ]


def test_structurally_different_equivalent_programs_both_remain_consistent():
    reversed_add = _candidate("commuted", (("add", (1, 0)),))
    scope = _scope()
    result = _run(scope, (_candidate(), reversed_add), _witnesses(scope, 4, 5))
    assert all(
        a.status is arb.DiagnosticStatus.CONSISTENT_ON_DECLARED_SCOPE for a in result.assessments
    )
    assert len(result.probes) == 2  # No discriminating input does not skip witnesses.


def test_thin_scope_does_not_claim_equivalence_beyond_it():
    scope = _scope((2, 2))
    result = _run(scope, witnesses=_witnesses(scope, 4))
    assert all(
        a.status is arb.DiagnosticStatus.CONSISTENT_ON_DECLARED_SCOPE for a in result.assessments
    )
    assert result.scope.interventions == scope.interventions
    assert result.serving_authority is False


@pytest.mark.parametrize("mode", ["absent", "missing", "empty", "budget", "single"])
def test_unobserved_cases_never_verify(mode):
    scope = _scope()
    witnesses = ()
    kwargs = {}
    candidates = None
    if mode == "missing":
        witnesses = _witnesses(scope, None, None)
    elif mode == "empty":
        scope = replace(scope, interventions=())
    elif mode == "budget":
        witnesses = _witnesses(scope, 4, 5)
        kwargs["max_probes"] = 0
    elif mode == "single":
        candidates = (_candidate(),)
    result = _run(scope, candidates, witnesses, **kwargs)
    assert all(a.status is arb.DiagnosticStatus.UNRESOLVED for a in result.assessments)
    if mode == "budget":
        assert result.prediction_count == 0
        assert result.untested == ("case:0", "case:1")


def test_partial_budget_preserves_untested_scope():
    scope = _scope()
    result = _run(scope, witnesses=_witnesses(scope, 4, 5), max_probes=1)
    assert result.assessments[0].status is arb.DiagnosticStatus.UNRESOLVED
    assert result.assessments[1].status is arb.DiagnosticStatus.CONTRADICTED
    assert result.untested == ("case:0",)


@pytest.mark.parametrize("silent", [True, False])
def test_silent_or_raising_prediction_never_verifies(monkeypatch, silent):
    def failed(*_args, **_kwargs):
        if silent:
            return SimpleNamespace(result=None)
        raise RuntimeError("observer could have failed in the same way")

    monkeypatch.setattr(arb, "execute_semantic_floor_program", failed)
    scope = _scope()
    for witnesses in (_witnesses(scope, 4, 5), _witnesses(scope, None, None)):
        result = _run(scope, witnesses=witnesses)
        assert all(
            a.status is arb.DiagnosticStatus.UNRESOLVED and len(a.silent) == 2
            for a in result.assessments
        )
        assert not any(p.contradicted for p in result.probes)
        assert result.probes[0].predictions[0].error == ("" if silent else "RuntimeError")


def test_floor_fuel_exhaustion_is_unresolved():
    scope = _scope()
    result = _run(scope, witnesses=_witnesses(scope, 4, 5), execution_fuel=1)
    assert all(a.status is arb.DiagnosticStatus.UNRESOLVED for a in result.assessments)
    assert all(p.error for record in result.probes for p in record.predictions)


def test_every_candidate_can_be_contradicted_without_forcing_a_winner():
    scope = _scope((2, 3))
    result = _run(scope, witnesses=_witnesses(scope, -1))
    assert all(a.status is arb.DiagnosticStatus.CONTRADICTED for a in result.assessments)


@pytest.mark.parametrize(
    "second,issue", [(6, "conflicting_observations"), (None, "incomplete_observations")]
)
def test_witness_conflict_or_silence_is_unresolved(second, issue):
    scope = _scope((2, 3))
    observations = _witnesses(scope, 5) + _witnesses(scope, second)
    for witnesses in (observations, tuple(reversed(observations))):
        result = _run(scope, witnesses=witnesses)
        assert all(a.status is arb.DiagnosticStatus.UNRESOLVED for a in result.assessments)
        assert result.probes[0].observation_issue == issue


@pytest.mark.parametrize("field", ["source", "tokens", "roles", "scope", "inputs"])
def test_witness_binding_changes_are_rejected_before_execution(monkeypatch, field):
    scope = _scope()
    witnesses = _witnesses(scope, 4, 5)
    changed = scope
    if field == "source":
        changed = replace(scope, source_text_sha256="a" * 64)
    elif field == "tokens":
        changed = replace(scope, source_token_ids=tuple(t + 1 for t in scope.source_token_ids))
    elif field == "roles":
        changed = replace(
            scope, roles=(replace(scope.roles[0], name="different role"), scope.roles[1])
        )
    elif field == "scope":
        changed = replace(scope, interpretation="Different source interpretation.")
    else:
        changed = replace(
            scope,
            interventions=(replace(scope.interventions[0], inputs=(9, 9)), scope.interventions[1]),
        )

    def forbidden(*_args, **_kwargs):
        pytest.fail("invalid binding reached execution")

    monkeypatch.setattr(arb, "compile_semantic_program_to_floor", forbidden)
    with pytest.raises(ValueError, match="source|scope"):
        _run(changed, witnesses=witnesses)


def test_ir_role_order_mismatch_is_rejected():
    scope = _scope()
    with pytest.raises(ValueError, match="ordered role spans"):
        _run(replace(scope, roles=tuple(reversed(scope.roles))))


@pytest.mark.parametrize(
    "declared,dependencies",
    [
        (arb.WitnessDerivation.CANDIDATE_DERIVED, ()),
        (arb.WitnessDerivation.SOURCE_OBSERVATION, ("3" * 64,)),
    ],
)
def test_direct_candidate_derivation_is_rejected_despite_observer_label(declared, dependencies):
    scope = _scope()
    witnesses = _witnesses(scope, 4, 5)
    provenance = arb.WitnessProvenance(
        "independent external oracle", "label is not authentication", declared, dependencies
    )
    with pytest.raises(ValueError, match="candidate-derived"):
        _run(scope, witnesses=tuple(replace(w, provenance=provenance) for w in witnesses))


def test_caller_can_lie_so_consistency_does_not_authenticate_provenance():
    scope = _scope()
    # These are multiplication outputs asserted as source observations.
    result = _run(scope, witnesses=_witnesses(scope, 4, 6))
    assert result.assessments[1].status is arb.DiagnosticStatus.CONSISTENT_ON_DECLARED_SCOPE
    assert result.provenance_authenticated is False
    assert result.serving_authority is False


def test_sequence_values_and_source_role_types():
    candidate = _candidate("lookup", (("at", (0, 1)),))
    scope = _scope(((3, 7, 11), 1), candidate=candidate)
    result = _run(scope, (candidate,), _witnesses(scope, 7))
    assert result.assessments[0].status is arb.DiagnosticStatus.CONSISTENT_ON_DECLARED_SCOPE
    with pytest.raises(ValueError, match="type differs"):
        replace(scope, interventions=(arb.Intervention("bad", (3, 1)),))


@pytest.mark.parametrize("value", [True, [1, 2], "<raised RuntimeError>", (True,), 1.0])
def test_observations_cannot_encode_exception_coincidence_or_mutable_values(value):
    scope = _scope((2, 3))
    with pytest.raises(ValueError):
        _witnesses(scope, value)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_probes": -1},
        {"max_probes": True},
        {"max_probes": 33},
        {"execution_fuel": 0},
        {"execution_fuel": 100001},
    ],
)
def test_invalid_budgets_are_rejected(kwargs):
    with pytest.raises(ValueError, match="budget|fuel"):
        _run(_scope(), **kwargs)


def test_result_preserves_exact_ir_identity_and_is_deterministic():
    scope = _scope()
    witnesses = _witnesses(scope, 4, 5)
    first = _run(scope, witnesses=witnesses)
    assert first == _run(scope, witnesses=witnesses)
    assert first.assessments[0].candidate.ir.receipt() == _candidate().ir.receipt()
    changed = replace(_candidate(), ir=replace(_candidate().ir, transducer_receipt_sha256="f" * 64))
    second = _run(scope, (changed, _candidate("mul")), witnesses)
    assert second != first  # Witnesses describe the source, not a candidate digest.


def test_witness_removal_and_shuffle_change_diagnostics_not_executability():
    scope = _scope()
    intact = _run(scope, witnesses=_witnesses(scope, 4, 5))
    removed = _run(scope)
    shuffled = _run(scope, witnesses=_witnesses(scope, 5, 4))
    assert [a.status for a in intact.assessments] == [
        arb.DiagnosticStatus.CONSISTENT_ON_DECLARED_SCOPE,
        arb.DiagnosticStatus.CONTRADICTED,
    ]
    assert all(a.status is arb.DiagnosticStatus.UNRESOLVED for a in removed.assessments)
    assert all(a.status is arb.DiagnosticStatus.CONTRADICTED for a in shuffled.assessments)
    assert [p.predictions for p in intact.probes] == [p.predictions for p in removed.probes]
    assert [p.predictions for p in intact.probes] == [p.predictions for p in shuffled.probes]
    assert intact.prediction_count == removed.prediction_count == shuffled.prediction_count == 4


def test_duplicate_witnesses_do_not_fill_missing_interventions():
    scope = _scope()
    witness = _witnesses(scope, 4, 5)[0]
    result = _run(scope, witnesses=(witness, witness))
    assert all(a.status is arb.DiagnosticStatus.UNRESOLVED for a in result.assessments)
    assert all(a.matched == ("case:0",) for a in result.assessments)


@pytest.mark.parametrize("kind", ["candidates", "interventions", "witnesses"])
def test_finite_collection_limits(kind):
    scope = _scope()
    candidates = (_candidate(),)
    witnesses = ()
    with pytest.raises(ValueError, match="bounded|limit"):
        if kind == "candidates":
            candidates = tuple(
                replace(candidates[0], name=str(i)) for i in range(arb.MAX_CANDIDATES + 1)
            )
        elif kind == "interventions":
            scope = replace(
                scope,
                interventions=tuple(
                    arb.Intervention(str(i), (2, 3)) for i in range(arb.MAX_INTERVENTIONS + 1)
                ),
            )
        else:
            witnesses = (_witnesses(scope, 4, 5)[0],) * (arb.MAX_WITNESSES + 1)
        arb.diagnose_source_programs(scope=scope, candidates=candidates, witnesses=witnesses)


@pytest.mark.parametrize("candidates", [(), (_candidate(), _candidate())])
def test_empty_or_duplicate_candidates_are_rejected(candidates):
    with pytest.raises(ValueError, match="candidate"):
        arb.diagnose_source_programs(scope=_scope(), candidates=candidates)


def test_scope_rejects_ambiguous_roles_and_duplicate_interventions():
    scope = _scope()
    with pytest.raises(ValueError, match="overlap"):
        replace(scope, roles=(scope.roles[0], replace(scope.roles[1], span=scope.roles[0].span)))
    with pytest.raises(ValueError, match="role names"):
        replace(scope, roles=(scope.roles[0], replace(scope.roles[1], name=scope.roles[0].name)))
    with pytest.raises(ValueError, match="intervention names"):
        replace(scope, interventions=(scope.interventions[0], scope.interventions[0]))


def test_witness_cannot_bind_an_intervention_outside_its_scope():
    scope = _scope()
    with pytest.raises(ValueError, match="outside"):
        replace(_witnesses(scope, 4, 5)[0], intervention=arb.Intervention("case:0", (9, 9)))


def test_a_silent_rival_is_not_eliminated_or_verified(monkeypatch):
    original = arb.execute_semantic_floor_program

    def sometimes(program, **kwargs):
        execution = original(program, **kwargs)
        return SimpleNamespace(result=None) if execution.result == 6 else execution

    monkeypatch.setattr(arb, "execute_semantic_floor_program", sometimes)
    scope = _scope()
    result = _run(scope, witnesses=_witnesses(scope, 4, 5))
    assert result.assessments[0].status is arb.DiagnosticStatus.CONSISTENT_ON_DECLARED_SCOPE
    assert result.assessments[1].status is arb.DiagnosticStatus.UNRESOLVED
    assert result.assessments[1].silent == ("case:1",)
    assert result.assessments[1].counterexamples == ()
