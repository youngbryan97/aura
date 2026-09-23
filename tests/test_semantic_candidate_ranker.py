"""The candidate selector consumes source evidence, not answer keys."""

from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as functional

from core.learning.procedure_induction import Instruction, Program
from core.learning.semantic_candidate_contrasts import (
    source_program_contrasts, source_program_factor_contrasts,
)
from core.learning.semantic_candidate_ranker import (
    ContextualProgramRanker, aggregate_program_scores, candidate_set_loss,
)
from core.learning.semantic_construction_folds import construction_folds
from core.learning.semantic_program_ir import TokenSpan
from core.learning.semantic_request_context import RequestContextConfig
from tools.compare_semantic_candidate_methods import _direct_choice, _portfolio_comparison
from tools.evaluate_semantic_candidate_ranker import (
    _evaluate, _factor_cases, _rankable, _rankable_or_none, _runtime_factor_cases,
    _source_runtime_factor_cases, _training_views,
)
from tools.materialize_semantic_candidate_training import _plan, _select_source_ids
from tools.replay_semantic_candidate_ranker_gap import _construction_counts


def test_mixed_methods_retain_disagreement_without_oracle_selection():
    programs = (Program(2, (Instruction("add", (0, 1)),)),
                Program(2, (Instruction("sub", (0, 1)),)))
    common = dict(programs=programs, keys=tuple(program.sha() for program in programs),
                  labels=(False, True), ranker_index=1, direct_index=0,
                  public_inputs=(7, 3), source="a" * 64,
                  transducer_receipt="b" * 64, ranker_receipt="c" * 64,
                  direct_receipt="d" * 64)
    rescued = _portfolio_comparison(**common, incumbent_present=False)
    assert rescued["portfolio_selected"] == "ranker"
    assert rescued["portfolio_correct"]
    assert rescued["portfolio_disagreements"] >= 1
    retained = _portfolio_comparison(**common, incumbent_present=True)
    assert retained["portfolio_selected"] == "incumbent"
    assert not retained["portfolio_correct"]


def test_gap_attribution_counts_coverage_separately_from_selection():
    rows = [
        {"construction": "role", "ranker_correct": True, "direct_correct": True,
         "portfolio_correct": False},
        {"construction": "role", "ranker_correct": False, "direct_correct": True,
         "portfolio_correct": True},
        {"construction": "alias", "ranker_correct": False, "direct_correct": False,
         "portfolio_correct": False},
    ]
    assert _construction_counts(rows) == {
        "alias": {"population": 1, "ranker_correct": 0, "direct_correct": 0,
                  "either_learned_correct": 0, "portfolio_correct": 0},
        "role": {"population": 2, "ranker_correct": 1, "direct_correct": 2,
                 "either_learned_correct": 2, "portfolio_correct": 1},
    }


def _programs():
    return (Program(2, (Instruction("add", (0, 1)),)),
            Program(2, (Instruction("sub", (0, 1)),)))


def _model():
    torch.manual_seed(29)
    return ContextualProgramRanker(RequestContextConfig(
        8, width=8, heads=2, layers=1, feature_scaling="unit_variance"))


def test_scores_complete_programs_and_learns_from_set_loss():
    model = _model()
    source = functional.normalize(torch.randn(4, 8), dim=-1)
    programs = _programs()
    scores = model(source, (TokenSpan(0, 1), TokenSpan(2, 3)),
                   ("integer", "integer"), programs,
                   operation_spans=((TokenSpan(1, 2),), (TokenSpan(1, 2),)))
    assert scores.shape == (2,)
    assert torch.isfinite(scores).all()
    loss = candidate_set_loss(scores, (False, True))
    loss.backward()
    assert model.context.project.weight.grad.abs().sum() > 0
    assert model.argument_step.weight_ih.grad.abs().sum() > 0
    assert model.operation.weight.grad.abs().sum() > 0
    assert model.key.weight.grad.abs().sum() > 0


def test_identity_binding_distinguishes_equal_value_source_mentions():
    torch.manual_seed(31)
    config = RequestContextConfig(8, width=8, heads=2, layers=1,
                                  position_mode="none", feature_scaling="unit_variance")
    model = ContextualProgramRanker(config, identity_bindings=True)
    source = functional.normalize(torch.ones(4, 8), dim=-1)
    spans = (TokenSpan(0, 1), TokenSpan(2, 3))
    programs = (Program(2, (Instruction("sub", (0, 1)),)),
                Program(2, (Instruction("sub", (1, 0)),)))
    anchors = ((TokenSpan(1, 2),),) * 2
    legacy = ContextualProgramRanker(config)
    legacy.load_state_dict({key: value for key, value in model.state_dict().items()
                            if not key.startswith("binding_")}, strict=True)
    scores = model(source, spans, ("integer", "integer"), programs,
                   operation_spans=anchors, cross_token=False)
    torch.testing.assert_close(scores, legacy(source, spans, ("integer", "integer"),
                                              programs, operation_spans=anchors,
                                              cross_token=False))
    assert torch.isclose(scores[0], scores[1])
    with torch.no_grad():
        model.binding_key.weight.copy_(torch.eye(config.width))
    scores = model(source, spans, ("integer", "integer"), programs,
                   operation_spans=anchors, cross_token=False)
    assert not torch.isclose(scores[0], scores[1])
    candidate_set_loss(scores, (True, False)).backward()
    assert model.binding_role.weight.grad.abs().sum() > 0
    assert model.binding_key.weight.grad.abs().sum() > 0


def test_argument_evidence_changes_same_program_only_when_its_channel_is_present():
    torch.manual_seed(37)
    config = RequestContextConfig(8, width=8, heads=2, layers=1,
                                  feature_scaling="unit_variance")
    model = ContextualProgramRanker(config, identity_bindings=True,
                                    argument_evidence=True,
                                    retain_evidence_variants=True).eval()
    source = functional.normalize(torch.randn(6, 8), dim=-1)
    program = Program(2, (Instruction("sub", (0, 1)),))
    programs = (program, program)
    kwargs = dict(operation_spans=((TokenSpan(2, 3),),) * 2,
                  argument_spans=(((TokenSpan(1, 2), TokenSpan(4, 5)),),
                                  ((TokenSpan(4, 5), TokenSpan(1, 2)),)),
                  definition_spans=(((TokenSpan(0, 1), TokenSpan(5, 6)),),) * 2)
    spans = (TokenSpan(0, 1), TokenSpan(5, 6))
    initial = model(source, spans, ("integer", "integer"), programs, **kwargs)
    torch.testing.assert_close(initial[0], initial[1])
    with torch.no_grad():
        model.evidence_key.weight[:, :config.width].copy_(torch.eye(config.width))
    observed = model(source, spans, ("integer", "integer"), programs, **kwargs)
    assert not torch.isclose(observed[0], observed[1])
    with torch.no_grad():
        model.evidence_key.weight.zero_()
    lesion = model(source, spans, ("integer", "integer"), programs, **kwargs)
    torch.testing.assert_close(lesion[0], lesion[1])
    with pytest.raises(ValueError, match="argument evidence"):
        model(source, spans, ("integer", "integer"), programs,
              operation_spans=kwargs["operation_spans"])


def test_legacy_ranker_checkpoint_has_no_identity_binding_parameters():
    model = _model()
    assert not any(key.startswith("binding_") for key in model.state_dict())
    restored = ContextualProgramRanker(model.config)
    restored.load_state_dict(model.state_dict(), strict=True)


def test_candidate_order_changes_scores_only_by_permutation():
    model = _model().eval()
    source = functional.normalize(torch.randn(4, 8), dim=-1)
    spans = (TokenSpan(0, 1), TokenSpan(2, 3))
    normal = model(source, spans, ("integer", "integer"), _programs())
    reversed_scores = model(source, spans, ("integer", "integer"), _programs()[::-1])
    torch.testing.assert_close(normal, reversed_scores.flip(0))


def test_operation_anchors_change_program_evidence_and_validate_shape():
    model = _model().eval()
    source = functional.normalize(torch.randn(5, 8), dim=-1)
    spans = (TokenSpan(0, 1), TokenSpan(3, 4))
    left = model(source, spans, ("integer", "integer"), _programs(),
                 operation_spans=((TokenSpan(1, 2),), (TokenSpan(1, 2),)))
    right = model(source, spans, ("integer", "integer"), _programs(),
                  operation_spans=((TokenSpan(2, 3),), (TokenSpan(2, 3),)))
    assert not torch.allclose(left, right)
    with pytest.raises(ValueError, match="anchors"):
        model(source, spans, ("integer", "integer"), _programs(),
              operation_spans=((TokenSpan(1, 2),),))


def test_source_evidence_affects_ranking_without_target():
    model = _model().eval()
    source = functional.normalize(torch.randn(4, 8), dim=-1)
    changed = source.clone()
    changed[3] = -changed[3]
    spans = (TokenSpan(0, 1), TokenSpan(2, 3))
    baseline = model(source, spans, ("integer", "integer"), _programs())
    observed = model(changed, spans, ("integer", "integer"), _programs())
    assert not torch.allclose(baseline, observed)


def test_invalid_graph_and_grounding_are_rejected():
    model = _model()
    source = functional.normalize(torch.randn(4, 8), dim=-1)
    with pytest.raises(ValueError, match="grammar"):
        model(source, (TokenSpan(0, 1), TokenSpan(2, 3)),
              ("integer", "integer"), (Program(2, (Instruction("add", (0, 2)),)),))
    with pytest.raises(ValueError, match="distinct"):
        model(source, (TokenSpan(0, 1), TokenSpan(0, 1)),
              ("integer", "integer"), _programs())
    with pytest.raises(ValueError, match="unit normalized"):
        model(source * 2, (TokenSpan(0, 1), TokenSpan(2, 3)),
              ("integer", "integer"), _programs())


def test_loss_rejects_no_verified_solution():
    with pytest.raises(ValueError, match="verified solution"):
        candidate_set_loss(torch.zeros(2), (False, False))
    with pytest.raises(ValueError, match="verified solution"):
        candidate_set_loss(torch.zeros(2), (1, 0))


def test_extra_evidence_paths_do_not_multiply_a_programs_prior_mass():
    scores = torch.zeros(4, requires_grad=True)
    keys, grouped, members = aggregate_program_scores(scores, ("right", "right", "right", "wrong"))
    assert keys == ("right", "wrong")
    assert members == ((0, 1, 2), (3,))
    torch.testing.assert_close(grouped, torch.zeros(2))
    loss = candidate_set_loss(scores, (True, True, True, False),
                              program_keys=("right", "right", "right", "wrong"))
    torch.testing.assert_close(loss, torch.tensor(0.6931472))
    loss.backward()
    assert scores.grad is not None and torch.isfinite(scores.grad).all()
    with pytest.raises(ValueError, match="contradictory"):
        candidate_set_loss(torch.zeros(2), (True, False),
                           program_keys=("same", "same"))


def test_proposal_view_separates_programs_from_diagnostic_labels():
    programs = _programs()
    candidates = [{"program": program.to_dict(), "program_sha256": program.sha(),
                   "operation_spans": [{"start": 1, "end": 2}]}
                  for program in programs]
    bank = {"candidates": candidates,
            "input_spans": [{"start": 0, "end": 1}, {"start": 2, "end": 3}],
            "selected_program_sha256": programs[0].sha()}
    item = SimpleNamespace(public_inputs=(2, 3))
    row = {"bank": bank, "diagnosis": {"comparisons": [
        {"program_sha256": programs[0].sha(), "status": "different"},
        {"program_sha256": programs[1].sha(), "status": "equivalent"}]}}
    (decoded, labels, keys), spans, kinds, anchors = _rankable(item, row)
    assert decoded == programs
    assert labels == (False, True)
    assert keys == tuple(program.sha() for program in programs)
    assert spans == (TokenSpan(0, 1), TokenSpan(2, 3))
    assert kinds == ("integer", "integer")
    assert anchors == ((TokenSpan(1, 2),), (TokenSpan(1, 2),))
    bank["selected_program_sha256"] = None
    (decoded, labels, _), _, _, _ = _rankable(item, row)
    assert decoded == programs
    assert labels == (False, True)


def test_runtime_evidence_variants_survive_program_hash_deduplication():
    program = _programs()[0]
    payload = program.to_dict()
    mentions = (({"start": 0, "end": 1}, {"start": 2, "end": 3}),)
    other = (({"start": 2, "end": 3}, {"start": 0, "end": 1}),)
    candidates = [{"program": payload, "program_sha256": program.sha(),
                   "operation_spans": [{"start": 1, "end": 2}],
                   "argument_spans": rows, "definition_spans": None,
                   "definition_provenance": "unavailable"}
                  for rows in (mentions, other)]
    item = SimpleNamespace(public_inputs=(3, 3))
    row = {"bank": {"schema": "aura.semantic_candidate_bank.v3",
                    "candidates": candidates,
                    "input_spans": [{"start": 0, "end": 1}, {"start": 2, "end": 3}],
                    "selected_program_sha256": program.sha()},
           "diagnosis": {"comparisons": [{"program_sha256": program.sha(),
                                           "status": "equivalent"}]}}
    assert len(_rankable(item, row)[0][0]) == 1
    case = _rankable(item, row, preserve_evidence=True)
    assert len(case[0][0]) == 2
    assert case[4][0] != case[4][1]
    assert case[0][1] == (True, True)
    row["bank"]["schema"] = "aura.semantic_candidate_bank.v2"
    with pytest.raises(ValueError, match="source-evidence"):
        _rankable(item, row, preserve_evidence=True)


def test_empty_ordinary_decode_stays_in_the_denominator_without_a_gradient():
    source = "a" * 64
    item = SimpleNamespace(public_inputs=(3, 3))
    row = {"bank": {"candidates": [], "input_spans": [],
                    "selected_program_sha256": None,
                    "limit_reason": "ordinary_decode_unavailable"}}
    assert _rankable_or_none(item, row) is None
    result = _evaluate(_model(), {source: item}, {source: row}, [source])
    assert result["population"] == 1
    assert result["ranker_evaluable"] == 0
    assert result["ranker_correct"] == 0
    assert result["rows"][0]["selected_correct"] is None
    row["bank"]["candidates"] = [{}]
    with pytest.raises((KeyError, ValueError)):
        _rankable_or_none(item, row)


def test_training_plan_requires_complete_train_cohort_and_fixed_bounds():
    examples = [SimpleNamespace(ir=SimpleNamespace(source_text_sha256=f"s{i}"), split="train")
                for i in range(3)]
    model = SimpleNamespace(training_receipt={"training_example_count": 3},
                            receipt_sha256="model", model_basis_sha256="basis")
    plan = _plan(model, examples, b"source", b"candidate", {}, "implementation",
                 max_charts=8, max_graphs=4, solve_seconds=2.)
    assert plan["source_ids"] == ("s0", "s1", "s2")
    assert plan["validation_used"] is False
    model.training_receipt["training_example_count"] = 4
    with pytest.raises(ValueError, match="complete frozen"):
        _plan(model, examples, b"source", b"candidate", {}, "implementation",
              max_charts=8, max_graphs=4, solve_seconds=2.)


def test_source_contrasts_are_typed_witnessed_and_deterministic():
    target, peer = _programs()
    source = "a" * 64
    first = source_program_contrasts(target, (5, 2), (peer,), source_sha256=source)
    second = source_program_contrasts(target, (5, 2), (peer,), source_sha256=source)
    assert first == second
    assert target in first and peer in first
    assert len({program.sha() for program in first}) == len(first)
    assert all(program.run((5, 2)) != target.run((5, 2))
               or any(program.run(values) != target.run(values)
                      for values in ((0, 1), (-3, 2), (2, 5)))
               for program in first if program != target)
    with pytest.raises(ValueError, match="source identity"):
        source_program_contrasts(target, (5, 2), (peer,), source_sha256="bad")


def test_source_factor_contrasts_include_witnessed_swap_and_later_operation():
    target = Program(3, (Instruction("count_of", (0, 1)), Instruction("sub", (3, 2))))
    inputs = ((8, 3, 5), 3, 2)
    rows = source_program_factor_contrasts(target, inputs, source_sha256="a" * 64)
    assert rows == source_program_factor_contrasts(target, inputs, source_sha256="a" * 64)
    assert target == rows[0]
    assert Program(3, (Instruction("count_of", (0, 1)), Instruction("sub", (2, 3)))) in rows
    assert Program(3, (Instruction("count_of", (0, 1)), Instruction("add", (3, 2)))) in rows
    assert len({program.sha() for program in rows}) == len(rows)
    assert all(any(program.run(values) != target.run(values)
                   for values in (inputs, ((2, 2, 3), 2, 4), ((5, 1), 1, 3)))
               for program in rows[1:])


def test_factor_training_cases_never_read_heldout_examples():
    target = Program(3, (Instruction("count_of", (0, 1)), Instruction("sub", (3, 2))))
    source = "a" * 64
    item = SimpleNamespace(public_inputs=((8, 3, 5), 3, 2),
                           ir=SimpleNamespace(to_program=lambda: target,
                                              input_spans=(TokenSpan(0, 1), TokenSpan(2, 3),
                                                           TokenSpan(4, 5)),
                                              instructions=(SimpleNamespace(
                                                  operation_span=TokenSpan(1, 2)),
                                                  SimpleNamespace(operation_span=TokenSpan(3, 4)))))
    cases = _factor_cases({source: item}, [source])
    assert len(cases[source][0][0]) >= 3
    assert sum(cases[source][0][1]) == 1
    assert _factor_cases({}, []) == {}
    assert _training_views(source, {source: cases[source]}, {}) == (cases[source],)


def test_runtime_factor_views_use_source_bank_spans_without_bank_labels():
    source = "a" * 64
    target = Program(2, (Instruction("sub", (0, 1)),))
    rival = Program(2, (Instruction("sub", (1, 0)),))
    spans = (TokenSpan(0, 1), TokenSpan(3, 4))
    item = SimpleNamespace(public_inputs=(3, 3), ir=SimpleNamespace(
        to_program=lambda: target, input_spans=spans,
        instructions=(SimpleNamespace(operation_span=TokenSpan(1, 2)),)))
    bank = (((target, rival), (False, False), (target.sha(), rival.sha())),
            spans, ("integer", "integer"),
            ((TokenSpan(2, 3),), (TokenSpan(1, 3),)))
    views = _runtime_factor_cases({source: item}, [source], {source: bank})[source]
    assert len(views) == 2
    assert [view[3][0] for view in views] == [bank[3][0], bank[3][1]]
    assert all(view[3][0] == view[3][1] for view in views)
    assert all(view[0][1][0] for view in views)
    assert _training_views(source, {source: bank}, {source: views}) == (bank, *views)
    with pytest.raises(ValueError, match="grounding"):
        _runtime_factor_cases({source: item}, [source], {source: (
            bank[0], (TokenSpan(0, 1), TokenSpan(2, 3)), bank[2], bank[3])})


def test_full_source_runtime_factors_use_only_declared_training_rows(monkeypatch):
    source = "a" * 64
    held = "b" * 64
    target = Program(2, (Instruction("sub", (0, 1)),))
    spans = (TokenSpan(0, 1), TokenSpan(3, 4))
    item = SimpleNamespace(split="train", public_inputs=(3, 3), ir=SimpleNamespace(
        source_text_sha256=source, to_program=lambda: target, input_spans=spans,
        instructions=(SimpleNamespace(operation_span=TokenSpan(1, 2)),)))
    view = SimpleNamespace(split="train", public_inputs=item.public_inputs,
                           ir=SimpleNamespace(source_text_sha256=source,
                                              to_program=lambda: target,
                                              input_spans=spans,
                                              instructions=(SimpleNamespace(
                                                  operation_span=TokenSpan(2, 3)),)))

    def acquire(_model, examples, *, max_operation_charts):
        assert examples == (item,)
        assert max_operation_charts == 2
        return (item, view), {"coverage": {"augmented": 1}}

    monkeypatch.setattr("core.learning.semantic_runtime_argument_views.runtime_argument_training_views",
                        acquire)
    cases, receipt = _source_runtime_factor_cases(None, {source: item, held: object()},
                                                   [source], max_charts=2)
    assert len(cases[source]) == 2
    assert [case[3][0] for case in cases[source]] == [
        (TokenSpan(1, 2),), (TokenSpan(2, 3),)]
    assert receipt["coverage"] == {"augmented": 1}
    assert held not in cases


def test_candidate_subset_is_selected_by_construction_without_labels():
    examples = [SimpleNamespace(ir=SimpleNamespace(source_text_sha256=f"s{i}"), split="train",
                                construction_id=f"group-{i // 2}", contrast_id=None)
                for i in range(6)]
    folds = construction_folds(examples, count=3)
    selected, selection = _select_source_ids(examples, folds, per_group=1)
    assert selected == ("s0", "s2", "s4")
    assert selection["labels_used_for_selection"] is False
    model = SimpleNamespace(training_receipt={"training_example_count": 6},
                            receipt_sha256="model", model_basis_sha256="basis")
    plan = _plan(model, examples, b"source", b"candidate", {}, "implementation",
                 max_charts=4, max_graphs=2, solve_seconds=2.,
                 selected_ids=selected, selection=selection)
    assert plan["pilot_only"] is True
    assert plan["source_population"] == 6
    bad = {**folds, "receipt_sha256": "bad"}
    with pytest.raises(ValueError, match="frozen construction"):
        _select_source_ids(examples, bad, per_group=1)


def test_mixed_training_keeps_two_grounded_candidate_views():
    spans = (TokenSpan(0, 1), TokenSpan(2, 3))
    kinds = ("integer", "integer")
    bank = ((_programs(), (True, False), tuple(p.sha() for p in _programs())),
            spans, kinds, ((TokenSpan(1, 2),),) * 2)
    contrasts = ((_programs()[::-1], (False, True),
                  tuple(p.sha() for p in _programs()[::-1])),
                 spans, kinds, ((TokenSpan(1, 2),),) * 2)
    assert _training_views("source", {"source": bank}, None) == (bank,)
    assert _training_views("source", {"source": bank}, {"source": contrasts}) == (
        bank, contrasts)
    changed = (contrasts[0], (TokenSpan(0, 1), TokenSpan(1, 2)), kinds, contrasts[3])
    with pytest.raises(ValueError, match="grounding"):
        _training_views("source", {"source": bank}, {"source": changed})
    permuted = (contrasts[0], spans[::-1], kinds, contrasts[3])
    assert _training_views("source", {"source": bank}, {"source": [permuted]},
                           allow_grounding_permutation=True) == (bank, permuted)


def test_direct_candidate_choice_uses_program_scores_not_labels():
    class Model:
        def score(self, _features, _spans, _kinds, program):
            return torch.tensor(2. if program.instructions[0].op == "sub" else 1.)

    spans = (TokenSpan(0, 1), TokenSpan(2, 3))
    features = functional.normalize(torch.randn(4, 8), dim=-1)
    assert _direct_choice(Model(), features, spans, ("integer", "integer"), _programs()) == 1
    assert _direct_choice(Model(), features, spans, ("integer", "integer"),
                          _programs()[::-1]) == 0
