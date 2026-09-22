"""The candidate selector consumes source evidence, not answer keys."""

from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as functional

from core.learning.procedure_induction import Instruction, Program
from core.learning.semantic_candidate_contrasts import source_program_contrasts
from core.learning.semantic_candidate_ranker import ContextualProgramRanker, candidate_set_loss
from core.learning.semantic_construction_folds import construction_folds
from core.learning.semantic_program_ir import TokenSpan
from core.learning.semantic_request_context import RequestContextConfig
from tools.compare_semantic_candidate_methods import _direct_choice, _portfolio_comparison
from tools.evaluate_semantic_candidate_ranker import _rankable, _training_views
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


def test_direct_candidate_choice_uses_program_scores_not_labels():
    class Model:
        def score(self, _features, _spans, _kinds, program):
            return torch.tensor(2. if program.instructions[0].op == "sub" else 1.)

    spans = (TokenSpan(0, 1), TokenSpan(2, 3))
    features = functional.normalize(torch.randn(4, 8), dim=-1)
    assert _direct_choice(Model(), features, spans, ("integer", "integer"), _programs()) == 1
    assert _direct_choice(Model(), features, spans, ("integer", "integer"),
                          _programs()[::-1]) == 0
